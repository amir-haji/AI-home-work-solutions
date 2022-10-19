from heapq import heapify, heappush, heappop
import math


def uniform_cost_search(s, g, adj_list):
    closed_cost_parent_list = {}
    frontier = []
    heapify(frontier)

    heappush(frontier, (0, s, -1))

    found = False
    while frontier:
        cost, node, parent = heappop(frontier)

        if node in closed_cost_parent_list:
            continue

        closed_cost_parent_list[node] = [cost, parent]

        if node == g:
            found = True
            break

        for adj_node, weight in adj_list[node]:
            dat = (weight + cost, adj_node, node)
            heappush(frontier, dat)

    if found:
        result = {}
        f = g
        while closed_cost_parent_list[f][1] != -1:
            result[f] = closed_cost_parent_list[f][0]
            f = closed_cost_parent_list[f][1]

        return found, result
    else:
        return found, {}


def theft_uniform_cost_search(thefts, goal, cars, adj_list):
    closed_list = {}
    frontier = []
    heapify(frontier)

    for x in thefts:
        dat = None
        if x in cars:
            dat = (0, x, 1)
        else:
            dat = (0, x, 0)

        heappush(frontier, dat)

    found = False
    answer = 0
    while frontier:

        cost, node, car = heappop(frontier)
        if node in closed_list:
            continue

        closed_list[node] = cost

        if node in cars:
            car = 1
        if node == goal:
            found = True
            answer = cost
            break

        for adj_node, weight in adj_list[node]:
            if car == 1:
                new_cost = cost + weight / 2
            else:
                new_cost = cost + weight

            dat1 = (new_cost, adj_node, car)
            heappush(frontier, dat1)

    if found:
        return answer
    else:
        return math.inf


n_test = int(input())

while n_test > 0:
    n, m = [int(x) for x in input().split()]
    adjacency_list = {}

    for i in range(m):
        data = [int(x) for x in input().split()]
        u, v, d = data
        if u in adjacency_list.keys():
            adjacency_list[u].append((v, d))
        else:
            adjacency_list[u] = [(v, d)]

        if v in adjacency_list.keys():
            adjacency_list[v].append((u, d))
        else:
            adjacency_list[v] = [(u, d)]

    T = int(input())
    t_cities = []
    c_cities = []
    if T != 0:
        t_cities = [int(x) for x in input().split()]
    C = int(input())
    if C != 0:
        c_cities = [int(x) for x in input().split()]

    start, goal = [int(x) for x in input().split()]
    ans, path_cost_list = uniform_cost_search(start, goal, adjacency_list)

    path_cost_list[start] = 0

    if not ans:
        print('Poor Tintin')
    else:
        cost = theft_uniform_cost_search(t_cities, goal, c_cities, adjacency_list)
        if cost < path_cost_list[goal]:
            print('Poor Tintin')
        else:
            print(path_cost_list[goal])
            goals = list(path_cost_list.keys())
            goals.reverse()
            print(len(goals))
            for x in goals:
                print(x, end=" ")
    n_test -= 1
