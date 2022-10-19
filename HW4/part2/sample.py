import itertools
import json
import os.path
import random
import matplotlib.pyplot as plt


class Graph:
    def __init__(self, nodes, edges):
        self.parents = None
        self.childs = None
        self.nodes = nodes
        self.edges = edges
        self.set_parents()
        self.set_childs()

    def set_parents(self):
        parents = {i: [] for i in self.nodes}
        for e in self.edges:
            parents[e[1]].append(e[0])

        self.parents = parents

    def set_childs(self):
        childs = {i: [] for i in self.nodes}

        for e in self.edges:
            childs[e[0]].append(e[1])

        self.childs = childs

    def recursive_sort(self, node, visited, stack):
        visited[node] = True

        for y in self.childs[node]:
            if not visited[y]:
                self.recursive_sort(y, visited, stack)

        stack.append(node)

    def topological_sort(self):
        visited = dict.fromkeys(self.nodes, False)
        stack = []

        for x in nodes:
            if not visited[x]:
                self.recursive_sort(x, visited, stack)
        ans = stack[::-1]
        return ans


class Factor:
    def __init__(self, cpt, index, parents):
        self.parents = parents
        self.index = index
        self.cpt = cpt

    def get_prob(self, values):
        query = []
        for x in self.parents:
            query.append(values[x])
        query = tuple(query)
        if type(self.cpt) == float:
            return self.cpt if values[self.index] else 1 - self.cpt
        else:
            return self.cpt[query] if values[self.index] else 1 - self.cpt[query]

    def get_prior_sample(self, values):
        query = []
        for x in self.parents:
            query.append(values[x])
        query = tuple(query)

        prob = 0
        if type(self.cpt) == float:
            prob = self.cpt
        else:
            prob = self.cpt[query]

        rand = random.uniform(0, 1)
        return 1 if rand <= prob else 0

    def get_reject_sample(self, values, qar):
        query = []
        for x in self.parents:
            query.append(values[x])
        query = tuple(query)

        prob = 0
        if type(self.cpt) == float:
            prob = self.cpt
        else:
            prob = self.cpt[query]

        rand = random.uniform(0, 1)
        ans = 1 if rand <= prob else 0

        if self.index in qar[1] and qar[1][self.index] != ans:
            return -1
        else:
            return ans

    def get_likelihood_sample(self, values, qar):
        query = []
        for x in self.parents:
            query.append(values[x])
        query = tuple(query)

        prob = 0
        if type(self.cpt) == float:
            prob = self.cpt
        else:
            prob = self.cpt[query]

        if self.index in qar[1]:
            if qar[1][self.index] == 0:
                prob = 1 - prob

            return qar[1][self.index], prob
        else:
            rand = random.uniform(0, 1)
            ans = 1 if rand <= prob else 0

            return ans, 1


def get_inputs(number):
    net_dir = 'inputs/' + str(number) + '/input.txt'
    query_dir = 'inputs/' + str(number) + '/q_input.txt'

    f = open(net_dir)
    n = int(f.readline())

    nodes = []
    edges = []
    factors = []

    for i in range(n):
        node = str(f.readline().replace('\n', ''))
        node = node.replace(' ', '')
        nodes.append(node)

        parents = f.readline().split()
        probs = None
        try:
            a = float(parents[0])
            probs = a
            parents = []
        except ValueError:
            for x in parents:
                edges.append((x, node))
            probs = {}
            for i in range(2 ** (len(parents))):
                a = [float(x) for x in f.readline().split()]
                d = tuple(int(x) for x in a[0:len(parents)])
                probs[d] = a[len(parents)]

        factors.append(Factor(probs, node, parents))

    g = Graph(nodes, edges)
    f.close()

    f = open(query_dir)
    query = json.loads(f.read())
    f.close()

    return g, nodes, factors, query


def make_table(factors, nodes):
    combinations = [list(x) for x in itertools.product([0, 1], repeat=len(nodes))]
    table = {}
    for x in combinations:
        ans = 1
        cpt = dict(zip(nodes, x))
        for y in factors:
            ans *= y.get_prob(cpt)
        table[tuple(x)] = ans
    return table


def get_conditional_table(query, table, nodes):
    new_table = {}
    for x in table.keys():
        equal = 0
        for y in query[1].keys():
            if x[nodes.index(y)] == query[1][y]:
                equal += 1
        if equal == len(query[1].keys()):
            new_table[x] = table[x]
    return new_table


def find_prob(query, table, nodes, method="exact"):
    t = table
    if len(query[1].keys()) != 0:
        t = get_conditional_table(query, table, nodes)

    ans = 0
    for x in t.keys():
        equal = 0
        for y in query[0].keys():
            if x[nodes.index(y)] == query[0][y]:
                equal += 1
        if equal == len(query[0].keys()):
            ans += t[x]
    if method == 'exact':
        if len(query[1].keys()) != 0:
            return ans/sum(t.values())
        else:
            return ans
    else:
        return ans/sum(t.values())


def get_prior_sampling(topo, nodes, factors):
    ans = {}
    for i in range(10000):
        q = {}
        for n in topo:
            fact = None
            for f in factors:
                if f.index == n:
                    fact = f
            q[n] = fact.get_prior_sample(q)

        lis = []
        for x in nodes:
            lis.append(q[x])
        lis = tuple(lis)
        if lis in ans:
            ans[lis] += 1
        else:
            ans[lis] = 1
    return ans


def get_reject_sampling(topo, nodes, factors, qar):
    ans = {}
    for i in range(10000):
        q = {}
        completed = True
        for n in topo:
            fact = None
            for f in factors:
                if f.index == n:
                    fact = f
            sample = fact.get_reject_sample(q, qar)
            if sample == -1:
                completed = False
                break
            else:
                q[n] = sample

        if not completed:
            continue
        else:
            lis = []
            for x in nodes:
                lis.append(q[x])
            lis = tuple(lis)
            if lis in ans:
                ans[lis] += 1
            else:
                ans[lis] = 1
    return ans


def get_likelihood_sampling(topo, nodes, factors, qar):
    ans = {}
    for i in range(10000):
        q = {}
        weight = 1
        for n in topo:
            fact = None
            for f in factors:
                if f.index == n:
                    fact = f
            sample, w = fact.get_likelihood_sample(q, qar)
            q[n] = sample
            weight *= w

        lis = []
        for x in nodes:
            lis.append(q[x])
        lis = tuple(lis)
        if lis in ans:
            ans[lis] += weight
        else:
            ans[lis] = weight
    return ans


def get_gibbs_sampling(nodes, factors, query):
    ans = {}
    state = {}
    non_evidence = []
    for x in nodes:
        if x in query[1]:
            state[x] = query[1][x]
        else:
            non_evidence.append(x)
            state[x] = random.choice([0, 1])
    i = 0
    for k in range(10000):
        fact = None
        for f in factors:
            if f.index == non_evidence[i]:
                fact = f

        sample = fact.get_prior_sample(state)
        new_state = state.copy()
        new_state[non_evidence[i]] = sample
        state = new_state

        lis = []
        for x in nodes:
            lis.append(state[x])
        lis = tuple(lis)

        if lis in ans.keys():
            ans[lis] += 1
        else:
            ans[lis] = 1

        i += 1
        if i == len(non_evidence):
            i = 0
    return ans


def create_text(*datas):
    if not os.path.exists('output'):
        os.mkdir('output')

    no = str(datas[0])
    dir = 'output/' + no + '.txt'
    file = open(dir, 'w')

    for i in range(len(datas[1])):
        data = ""
        for j in range(1, 6):
            data = data + str(datas[j][i]) + " "
        data = data + "\n"
        file.write(data)

    file.close()


def create_plot(*datas):
    no = str(datas[0])
    x = [i for i in range(1, len(datas[1]) + 1)]

    plt.plot(x, datas[1], 'o-r', label='prior')
    plt.plot(x, datas[2], 'o-b', label='reject')
    plt.plot(x, datas[3], 'o-g', label='likelihood')
    plt.plot(x, datas[4], 'o-k', label='gibbs')

    plt.ylabel('MAE')
    plt.xlabel('#Q')

    plt.legend(loc='upper right')

    dir = os.path.abspath(__file__)
    dir = dir.replace('sample.py', '')
    con = 'output/' + str(no) + '.png'

    plt.savefig(os.path.join(dir, con))
    plt.close()

#######################################################################
#######################################################################


fine_no = [int(x) for x in os.listdir('inputs')]

for number in fine_no:

    g, nodes, factors, query = get_inputs(number)

    table = make_table(factors, nodes)

    exact_inference = []
    for x in query:
        exact_inference.append(round(find_prob(x, table, nodes), 5))

    topo_sorted = g.topological_sort()

    prior_table = get_prior_sampling(topo_sorted, nodes, factors)
    prior_results = []
    for i in range(len(query)):
        prior_results.append(round(abs(find_prob(query[i], prior_table, nodes, 'prior') - exact_inference[i]), 5))

    reject_results = []
    for i in range(len(query)):
        reject_table = get_reject_sampling(topo_sorted, nodes, factors, query[i])
        reject_results.append(round(abs(find_prob(query[i], reject_table, nodes, 'prior') - exact_inference[i]), 5))

    likelihood_results = []
    for i in range(len(query)):
        likelihood_table = get_likelihood_sampling(topo_sorted, nodes, factors, query[i])
        likelihood_results.append(
            round(abs(find_prob(query[i], likelihood_table, nodes, 'prior') - exact_inference[i]), 5))

    gibbs_results = []
    for i in range(len(query)):
        gibbs_table = get_gibbs_sampling(nodes, factors, query[i])
        gibbs_results.append(round(abs(find_prob(query[i], gibbs_table, nodes, 'prior') - exact_inference[i]), 5))

    create_text(number, exact_inference, prior_results, reject_results, likelihood_results, gibbs_results)

    create_plot(number, prior_results, reject_results, likelihood_results, gibbs_results)




