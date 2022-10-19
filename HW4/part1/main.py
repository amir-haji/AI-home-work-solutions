import itertools


class Graph:
    def __init__(self, n, edges):
        self.parents = None
        self.childs = None
        self.n = n
        self.nodes = [i for i in range(1, n+1)]
        self.edges = edges
        self.set_parents()
        self.set_childs()

    def set_parents(self):
        parents = {i: [] for i in range(1, self.n + 1)}
        for e in self.edges:
            parents[e[1]].append(e[0])

        self.parents = parents

    def set_childs(self):
        childs = {i: [] for i in range(1, self.n + 1)}
        for e in self.edges:
            childs[e[0]].append(e[1])

        self.childs = childs

    def recursive_sort(self, node, visited, stack):
        visited[node] = True

        for y in self.childs[node]:
            if not visited[y]:
                self.recursive_sort(y, visited, stack)

        stack.append(node)

    def topological_reversed_sort(self):
        visited = dict.fromkeys(self.nodes, False)
        stack = []

        for x in self.nodes:
            if not visited[x]:
                self.recursive_sort(x, visited, stack)

        return stack


class Factor:
    def __init__(self, probs, index, parents):
        if parents is None:
            self.parents = []
        else:
            parents.sort()
            self.parents = parents
        self.index = index
        self.list = [self.index] + self.parents
        self.cpt = {}
        self.create_cpt(probs)

    def create_cpt(self, probs):

        perms = []
        for com in itertools.combinations_with_replacement([True, False], len(self.parents) + 1):
            for rep in itertools.permutations(com):
                if list(rep) not in perms:
                    perms.append(list(rep))

        perms.sort(reverse=True)

        for k in range(2 ** len(self.parents)):
            self.cpt[tuple(perms[k])] = probs[k]

        for k in range(2 ** len(self.parents), 2 ** (len(self.parents) + 1)):
            self.cpt[tuple(perms[k])] = 1 - probs[k - 2 ** len(self.parents)]

    def delete_non_evidence(self, query):
        check = {}
        for x in self.list:
            if x in query[1]:
                check[x] = query[1][x]

        deleteds = []

        for x in self.cpt.keys():
            res = False
            for y in check:
                if x[self.list.index(y)] != check[y]:
                    res = True
                    break

            if res:
                deleteds.append(x)

        for x in deleteds:
            del self.cpt[x]

    def get_prob(self, value):
        lis = []

        for x in self.list:
            lis.append(value[x])

        lis = tuple(lis)

        return self.cpt[lis]


def find_actives(start, end, graph, givens):
    l = givens.copy()
    ancestors = []
    while l:
        node = l.pop()
        ancestors = ancestors + [node]
        if not node in ancestors:
            l = l + graph.parents[node]

        ancestors = list(dict.fromkeys(ancestors))

    step = [(start, True)]
    reachable, visited = [], []
    while step:
        node, dir = step.pop()
        if not (node, dir) in visited:
            visited = visited + [(node, dir)]
            if node not in givens:
                reachable = reachable + [node]

            if (not node in givens) and dir:
                for i in graph.parents[node]:
                    step = step + [(i, True)]
                for i in graph.childs[node]:
                    step = step + [(i, False)]

            if not dir:
                if not node in givens:
                    for i in graph.childs[node]:
                        step = step + [(i, False)]
                if node in ancestors:
                    for i in graph.parents[node]:
                        step = step + [(i, True)]

    return list(dict.fromkeys(reachable))


def join_sum_factors(targets, var, query):
    #community
    new_list = []
    for x in targets:
        new_list = new_list + x.list

    new_list = list(dict.fromkeys(new_list))

    # get hiddens
    new_list.sort()
    new_list.remove(var)

    result = Factor([1], 1, [])
    result.list = new_list
    result.cpt.clear()

    hiddens = list(filter(lambda x: x not in query[1] and x != var, new_list))

    states = {x: query[1][x] for x in query[1]}

    combinations = [list(x) for x in itertools.product([0, 1], repeat=len(hiddens))]

    for x in combinations:
        dict1 = dict(zip(hiddens, x))
        q = {** states, **dict1}
        w1 = 1
        w2 = 1
        for y in targets:
            q[var] = 1
            w1 *= y.get_prob(q)
            q[var] = 0
            w2 *= y.get_prob(q)

        del q[var]

        w1 += w2
        tup = []
        for x in new_list:
            tup.append(q[x])

        tup = tuple(tup)
        result.cpt[tup] = w1

    return result


def find_ans(targets, query):
    #community
    new_list = []
    for x in targets:
        new_list = new_list + x.list

    new_list = list(dict.fromkeys(new_list))

    # get hiddens
    new_list.sort()

    result = Factor([1], 1, [])
    result.list = new_list
    result.cpt.clear()

    states = {x: query[1][x] for x in query[1]}
    q = states.copy()
    keys = list(query[0].keys())
    main_query = keys[0]
    for i in range(2):
        q[main_query] = i
        w = 1

        for y in targets:
            w *= y.get_prob(q)

        tup = []
        for x in new_list:
            tup.append(q[x])

        tup = tuple(tup)
        result.cpt[tup] = w

    s = sum(result.cpt.values())
    for x in result.cpt:
        result.cpt[x] /= s

    return result


def delete_hidden_var(factors, var, query):
    targets = []
    for f in factors:
        if var in f.list:
            targets.append(f)

    for x in targets:
        factors.remove(x)

    new_factor = join_sum_factors(targets, var, query)
    factors.append(new_factor)
    return factors


def variable_elimination_table(factors, sorted, query):
    hiddens = list(filter(lambda x: x not in query[1] and x not in query[0], sorted))
    for x in hiddens:
        factors = delete_hidden_var(factors, x, query)

    return factors


n = int(input())
edges = []
factors1 = []
factors2 = []

vars = [i for i in range(1, n + 1)]
for i in range(1, n + 1):
    parents = [int(x) for x in input().split()]
    for x in parents:
        edges.append((x, i))

    cpt = [float(x) for x in input().split()]

    factors1.append(Factor(cpt, i, parents))
    factors2.append(Factor(cpt, i, parents))

givens = {}
lis = [x for x in input().split(',')]
for y in lis:
    a, b = [int(x) for x in y.split('->')]
    givens[a] = b

start, end = [int(x) for x in input().split()]

g = Graph(n, edges)
for x in givens:
    vars.remove(x)

ans = find_actives(start, end, g, list(givens.keys()))

if end in ans:
    print('dependent')
else:
    print('independent')

query1 = [{}, givens.copy()]
query2 = [{}, givens.copy()]

query1[0][start] = 1
query2[0][end] = 1

for x in factors1:
    x.delete_non_evidence(query1)

for x in factors2:
    x.delete_non_evidence(query2)


sorted = g.topological_reversed_sort()

factors1 = variable_elimination_table(factors1, sorted, query1)
factor1 = find_ans(factors1, query1)

factors2 = variable_elimination_table(factors2, sorted, query2)
factor2 = find_ans(factors2, query2)


for x in query1[0]:
    query1[1][x] = query1[0][x]

for x in query2[0]:
    query2[1][x] = query2[0][x]

print(round(factor1.get_prob(query1[1]), 2))

print(round(factor2.get_prob(query2[1]), 2))
