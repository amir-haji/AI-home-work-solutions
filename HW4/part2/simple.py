import os.path
import random
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def gaussian_pdf(x, mean, var):
    a = 1 / math.sqrt(2 * math.pi * var)
    b = math.e ** (-((x - mean)**2)/(2*var))
    return a * b


def geometric_pdf(x, p):
    return p * (1-p)**(x - 1)


def exponential_pdf(x, l):
    return math.e**(-l*x)


def pdf_1(x):
    return 0.3 * gaussian_pdf(x, 4, 2) + 0.3 * gaussian_pdf(x, 3, 2) + 0.4 * exponential_pdf(x, 0.01)


def pdf_2(x):
    return 0.2 * gaussian_pdf(x, 0, 10) + 0.2 * gaussian_pdf(x, 20, 15) + 0.3 * gaussian_pdf(x, -10, 8) + 0.3 * gaussian_pdf(x, 50, 25)


def pdf_3(x):
    return 0.2 * geometric_pdf(x, 0.1) + 0.2 * geometric_pdf(x, 0.5) + 0.2 * geometric_pdf(x, 0.3) + 0.4 * geometric_pdf(x, 0.04)


def get_bernouli_sample():
    return random.randint(0, 1)


def bernouli(p):
    rand = random.uniform(0, 1)
    if 0 <= rand < p:
        return 1
    else:
        return 0


def geometric(p):
    i = 0
    result = 0
    while not result:
        result = bernouli(p)
        i += 1

    return i


def exponential(l):
    rand = random.uniform(0, 1)
    return (-1 / l) * math.log(1 - rand)


def gausian_sample(e, var, n=2500):
    normal = 2 * math.sqrt(n) * ((sum(get_bernouli_sample() for _ in range(n)) / n) - 0.5)
    return math.sqrt(var) * normal + e


def get_sample_pdf1():
    rand = random.uniform(0, 1)

    if 0 <= rand < 0.3:
        return gausian_sample(4, 2)
    elif 0.3 <= rand < 0.6:
        return gausian_sample(3, 2)
    else:
        return geometric(0.01)

    # return 0.3 * gausian_sample(4, 2) + 0.3 * gausian_sample(3, 2) + 0.4 * geometric(0.01)


def get_sample_pdf2():
    rand = random.uniform(0, 1)
    if 0 <= rand < 0.2:
        return gausian_sample(0, 10)
    elif 0.2 <= rand < 0.4:
        return gausian_sample(20, 15)
    elif 0.4 <= rand < 0.7:
        return gausian_sample(-10, 8)
    else:
        return gausian_sample(50, 25)
    # return 0.2 * gausian_sample(0, 10) + 0.2 * gausian_sample(20, 15) + 0.3 * gausian_sample(-10, 8) + 0.3 * gausian_sample(50, 25)


def get_sample_pdf3():
    rand = random.uniform(0, 1)
    if 0 <= rand < 0.2:
        return geometric(0.1)
    elif 0.2 <= rand < 0.4:
        return geometric(0.5)
    elif 0.4 <= rand < 0.6:
        return geometric(0.3)
    else:
        return geometric(0.04)


def create_txt(samples):
    if not os.path.exists('part1'):
        os.mkdir('part1')

    file = open('part1/log.txt', 'w')
    for i in range(1, 4):
        e = np.round(np.mean(samples[i-1]), 4)
        std = np.round(np.std(samples[i-1]), 4)

        st = ''
        st = str(i) + ' ' + str(e) + ' ' + str(std) + '\n'
        file.write(st)

    file.close()


def pdf_plot(*lists):
    for i in range(1, 4):
        plt.plot(lists[i - 1][0], lists[i - 1][1])
        dir = 'part1/pdf'
        dir = dir + str(i) + '.png'
        plt.savefig(dir)
        plt.close()


def hist_plot(lists):
    for i in range(1, 4):
        df = pd.DataFrame(data=lists[i-1], columns=['a'])

        ax = df['a'].plot.hist(bins=100, density=True, edgecolor='w', linewidth=0.5)

        xlim = ax.get_xlim()

        df['a'].plot.density(color='k', alpha=0.5, ax=ax)

        ax.set_xlim(xlim)
        dir = 'part1/pdf'
        dir = dir + str(i) + '_sample.png'
        plt.savefig(dir)
        plt.close()


samples = [[], [], []]

for i in range(10000):
    samples[0].append(get_sample_pdf1())
    samples[1].append(get_sample_pdf2())
    samples[2].append(get_sample_pdf3())

create_txt(samples)

data_pdf_1 = [[i for i in range(0, 800)], []]
data_pdf_2 = [[i for i in range(-70, 70)], []]
data_pdf_3 = [[i for i in range(0, 800)], []]

for i in range(0, 800):
    data_pdf_1[1].append(pdf_1(i))
    data_pdf_3[1].append(pdf_3(i))

for i in range(-70, 70):
    data_pdf_2[1].append(pdf_2(i))


pdf_plot(data_pdf_1, data_pdf_2, data_pdf_3)

hist_plot(samples)

