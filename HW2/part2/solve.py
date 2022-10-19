import hardest_game
import random
import math
import matplotlib.pyplot as plt

def play_game_AI(str, map_name='map1.txt'):
    game = hardest_game.Game(map_name=map_name, game_type='AI').run_AI_moves_graphic(moves=str)
    return game


def simulate(str, map_name='map1.txt'):
    game = hardest_game.Game(map_name=map_name, game_type='AI').run_AI_moves_no_graphic(moves=str)
    return game


def run_whole_generation(list_of_strs, N, map_name='map1.txt'):
    game = hardest_game.Game(map_name=map_name, game_type='AIS').run_generation(list_of_moves=list_of_strs, move_len=N)
    return game


def play_human_mode(map_name='map1.txt'):
    hardest_game.Game(map_name=map_name, game_type='player').run_player_mode()



def score(st, maps):
    
    game = simulate(st, maps)
    value = 0
    if game.hasWon:
        value += 50

    if not game.hasDied:
        for g in game.goals:
            if g[1]:
                value += 10
        dis = abs(game.player.x - game.end.x) + abs(game.player.y - game.end.y)
        try:
            value += 1000 / dis
        except ZeroDivisionError:
            value += 5
        
    else:
        value = 0

    return value
    


def fitness(generation, maps):
    fit = []
    for x in generation:
        sc = score(x, maps)
        fit.append(sc)

    return fit


def select(fit_list):
    sum = 0
    for x in fit_list:
        sum += x

    rand = random.uniform(0, sum)
    index = 0
    while rand > 0:
        rand -= fit_list[index]
        index += 1
    index -= 1
    return index


def select_2(fit_list):
    return fit_list.index(max(fit_list))


def crossover(st1, st2):
    i = random.randint(1, len(st1) - 2)
    c1 = st1[:i] + st2[i:]
    c2 = st2[:i] + st1[i:]
    return c1, c2


def mutate(st, p_mutate, chars):
    rand = random.uniform(0, 1)
    if rand > p_mutate:
        return st

    rand2 = random.randint(0, len(st) - 1)
    index = random.randint(0, 3)
    st_list = list(st)
    st_list[rand2] = chars[index]
    ans = "".join(st_list)
    return ans


def append_genome(st, chars):
    ans = []
    for x in chars:
        st1 = st
        st1 = st1 + x
        ans.append(st1)

    return ans




def solve_problem(maps='map2.txt'):
    chars = ['w', 'd', 's', 'a', 'x']
    population = []
    generation_results = []
    p_mut = 0.1
    for i in range(4):
        st = ""
        for j in range(3):
            index = random.randint(0, len(chars) - 1)
            st = st + chars[index]
        population.append(st)

    counter = 0
    while True:
        counter += 1

        fit = fitness(population, maps)
        g = play_game_AI(population[fit.index(max(fit))], maps)
        generation_results.append(max(fit))

        if g.hasWon:
            break
        
        

        new_population = []

        for x in append_genome(population[select_2(fit)], chars):
            if score(x, maps) != 0:
                new_population.append(x)

        while len(new_population) <= 20:
            parent1 = population[select(fit)]
            parent2 = population[select(fit)]
            child1, child2 = crossover(parent1, parent2)
            child1 = mutate(child1, p_mut, chars)
            child2 = mutate(child2, p_mut, chars)

            for x in append_genome(child1, chars):
                if score(x, maps) != 0:
                    new_population.append(x)

            for x in append_genome(child2, chars):
                if score(x, maps) != 0:
                    new_population.append(x)

        population = new_population

    print(generation_results)
    fit = fitness(population, maps)
    ans = population[fit.index(max(fit))]
    plt.xlabel('generation number')
    plt.ylabel('maximum fitness')
    plt.title('maximum fitness per generation for map2.txt')
    plt.plot(generation_results)
    plt.show()
    
    print(ans)
    


solve_problem()
