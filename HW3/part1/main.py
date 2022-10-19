import time
import copy

def print_grid(arr):
    for i in range(9):
        for j in range(9):
            print(arr[i][j], end=" ")
        print()


def find_empty_location(arr, l, values):
    # values = dict(sorted(values.items(), key=lambda x: len(x[1])))
    for x in values:
        row = x // 9
        col = x % 9

        if arr[row][col] == 0:
            l[0] = row
            l[1] = col
            return True

    return False


def lcv(arr, values, x, value):
    neigh = neighbors(arr, x)

    c = 0
    for k in neigh:
        c += len(values[k])
        if value in values[k]:
            c -= 1

    return c


def lcv_sort(arr, values, y):
    values[y] = sorted(values[y], key=lambda x: lcv(arr, values, y, x), reverse=True)

    return values


def used_in_row(arr, row, num):
    for i in range(9):
        if arr[row][i] == num:
            return True
    return False


def used_in_col(arr, col, num):
    for i in range(9):
        if arr[i][col] == num:
            return True
    return False


def used_in_box(arr, row, col, num):
    for i in range(3):
        for j in range(3):
            if arr[i + row][j + col] == num:
                return True
    return False


def check_location_is_safe(arr, row, col, num):
    return not used_in_row(arr, row, num) and not used_in_col(arr, col, num) and not used_in_box(arr, row - row % 3, col - col % 3, num)


def solve_sudoku(arr, values):
    l = [0, 0]

    if not find_empty_location(arr, l, values):
        return True

    row = l[0]
    col = l[1]
    x = 9 * row + col
    values = lcv_sort(arr, values, x)
    for num in values[x]:

        arr[row][col] = num

        queue = []

        for k in neighbors(arr, x):
            if [k, x] not in queue:
                queue.append([k, x])

        v1 = copy.deepcopy(values)
        v1[x] = [num]
        new_values, ans = ac_3(arr, queue, v1)

        if not ans:
            arr[row][col] = 0
        else:
            if solve_sudoku(arr, new_values):
                return True
            arr[row][col] = 0

    return False


def initial_queue(arr, cells):
    queue = []
    for x in cells:
        for k in neighbors(arr, x):
            if [k, x] not in queue:
                queue.append([k, x])
    return queue


def ac_3(arr, queue, values):
    ans = True
    while queue:
        x_i, x_j = queue.pop()
        values, deleted = revise(arr, values, x_i, x_j)

        if deleted:
            if len(values[x_i]) == 0:
                ans = False
                break
            else:
                for k in get_neighbors(x_i, x_j):
                    if [k, x_i] not in queue:
                        queue.append([k, x_i])

    return values, ans


def revise(arr, values, x, y):
    deleted = False
    des = []

    for a in values[x]:
        counter = 0
        for b in values[y]:
            if a != b:
                counter += 1
                break

        if counter == 0:
            des.append(a)
            deleted = True

    if arr[x // 9][x % 9] == 0:
        for z in des:
            values[x].remove(z)

    return values, deleted


def get_neighbors(x, y):
    ans = []

    row = x // 9
    col = x % 9
    for i in range(9):
        if 9 * row + i != y and 9 * row + i != x and 9 * row + i not in ans:
            ans.append(9 * row + i)

    for i in range(9):
        if 9 * i + col != y and 9 * i + col != x and 9 * i + col not in ans:
            ans.append(9 * i + col)

    row -= row % 3
    col -= col % 3

    for i in range(row, row + 3):
        for j in range(col, col + 3):
            if 9 * i + j != y and 9 * i + j != x and 9 * i + j not in ans:
                ans.append(9 * i + j)

    return ans


def neighbors(arr, x):
    ans = []

    row = x // 9
    col = x % 9
    for i in range(9):
        if 9 * row + i != x and 9 * row + i not in ans and arr[row][i] == 0:
            ans.append(9 * row + i)

    for i in range(9):
        if 9 * i + col != x and 9 * i + col not in ans and arr[i][col] == 0:
            ans.append(9 * i + col)

    row -= row % 3
    col -= col % 3

    for i in range(row, row + 3):
        for j in range(col, col + 3):
            if 9 * i + j != x and 9 * i + j not in ans and arr[i][j] == 0:
                ans.append(9 * i + j)

    return ans


def delete_value_neighbors(arr, row, col, values, num):
    for i in range(9):
        if arr[row][i] == 0:
            if i != col and num in values[9 * row + i]:
                values[9 * row + i].remove(num)

    for i in range(9):
        if arr[i][col] == 0:
            if i != row and num in values[9 * i + col]:
                values[9 * i + col].remove(num)

    x = 9 * row + col
    row -= row % 3
    col -= col % 3

    for i in range(row, row + 3):
        for j in range(col, col + 3):
            if 9 * i + j != x and num in values[9 * i + j]:
                values[9 * i + j].remove(num)

    return values


if __name__ == "__main__":

    empty_cell_values = {}
    grid = [[0 for x in range(9)] for y in range(9)]
    non_zero_cells = []


    for i in range(9):
        list = [x for x in input().split()]

        for j in range(9):
            if list[j] == '.':
                grid[i][j] = 0
                empty_cell_values[9 * i + j] = [x for x in range(1, 10)]

            else:
                grid[i][j] = int(list[j])
                empty_cell_values[9 * i + j] = [int(list[j])]
                non_zero_cells.append(9 * i + j)

    queue = initial_queue(grid, non_zero_cells)
    values, ans = ac_3(grid, queue, empty_cell_values)
    ''''
    new_values = {}
    for x in values:
        if x not in non_zero_cells:
            new_values[x] = values[x]
    '''''
    new_values = dict(sorted(values.items(), key=lambda x: len(x[1])))

    if solve_sudoku(grid, new_values):
        print_grid(grid)

