import math
from heapq import heapify,heappush, heappop


def create_map(n, m):
    maps = []
    for i in range(n+2):
        maps.append([0] * (m+2))

    for i in range(m + 2):
        maps[0][i] = -1
        maps[n+1][i] = -1

    for i in range(n + 2):
        maps[i][0] = -1
        maps[i][m+1] = -1

    tile_numbers = [x for x in range(2, n * m + 1)]

    return maps, tile_numbers


def print_map(maps):
    for i in range(n + 2):
        for j in range(m + 2):
            print(maps[i][j], end=" ")
        print()
    print("**")


def copy_map(maps):
    new = []
    for x in maps:
        new.append(x.copy())

    return new


n, m = [int(x) for x in input().split()]

tile_mains = []

data1 = [int(x) for x in input().split()]
mini = min(data1)
tile_mains.append(data1)

for i in range(n*m - 1):
    data = [int(x) for x in input().split()]
    mini = min(mini, min(data))
    tile_mains.append(data)


# f - map - g - h - tiles
frontier = []
heapify(frontier)
for i in range(1, n + 1):
    for j in range(1, m + 1):
        maps, numbers = create_map(n, m)
        maps[i][j] = 1
        heappush(frontier, (0, maps, 0, (n*m - 1) * mini, numbers))

answers = []
heapify(answers)

while len(frontier):
    f, maps, g, h, tiles = heappop(frontier)

    if h == 0:
        heappush(answers, f)
        continue

    e_map = None
    min_f = math.inf
    for tile_no in tiles:
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                if maps[i][j] == 0:
                    places = [[i-1, j], [i, j+1], [i+1, j], [i, j-1]]

                    for d in range(4):
                        x, y = places[d]
                        if maps[x][y] > 0:
                            tile1 = maps[x][y]
                            c = (d + 2) % 4
                            if tile_mains[tile1-1][c] == tile_mains[tile_no - 1][d]:
                                cost = tile_mains[tile_no - 1][d]
                                new_map = copy_map(maps)
                                new_map[i][j] = tile_no
                                new_tiles = tiles.copy()
                                new_tiles.remove(tile_no)
                                new_g = g + cost
                                new_h = h - mini
                                new_f = new_g + new_h
                                dat = (new_f, new_map, new_g, new_h, new_tiles)
                                if new_f < min_f:
                                    min_f = new_f
                                    e_map = [dat]
                                elif new_f == min_f:
                                    e_map.append(dat)

    if e_map is not None:
        for x in e_map:
            heappush(frontier, x)


print(answers[0])
