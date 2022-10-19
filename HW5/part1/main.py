import numpy as np
import math


def gaussian_pdf(x, mean, var):
    # a = 1 / math.sqrt(2 * math.pi * var)
    b = math.e ** (-((x - mean)**2)/(2*var))
    return b


def get_weights(samples, oly_evidences, ars_evidences, pav_evidences, asc_evidences, i):
    oly_x, oly_y = -133, 18
    ars_x, ars_y = -121, -9
    pav_x, pav_y = -113, 1
    asc_x, asc_y = -104, 12

    weights = np.array([1] * len(samples), dtype='f8')
    for j in range(len(samples)):
        dif_oly = -np.sqrt((samples[j, 0] - oly_x) ** 2 + (samples[j, 1] - oly_y) ** 2) + oly_evidences[i]
        dif_ars = -np.sqrt((samples[j, 0] - ars_x) ** 2 + (samples[j, 1] - ars_y) ** 2) + ars_evidences[i]
        dif_pav = -np.sqrt((samples[j, 0] - pav_x) ** 2 + (samples[j, 1] - pav_y) ** 2) + pav_evidences[i]
        dif_asc = -np.sqrt((samples[j, 0] - asc_x) ** 2 + (samples[j, 1] - asc_y) ** 2) + asc_evidences[i]

        weights[j] *= gaussian_pdf(dif_oly, 2, 1)
        weights[j] *= gaussian_pdf(dif_ars, 2, 1)
        weights[j] *= gaussian_pdf(dif_pav, 2, 1)
        weights[j] *= gaussian_pdf(dif_asc, 2, 1)

    # weights.sort()
    # weights = weights[::-1]

    return weights


evidences = {}
for _ in range(4):
    name = input()
    inputs = []
    for i in range(20):
        inputs.append(float(input()))

    evidences[name] = inputs


oly_evidences = np.array(evidences['oly'])
ars_evidences = np.array(evidences['ars'])
pav_evidences = np.array(evidences['pav'])
asc_evidences = np.array(evidences['asc'])

samples_x = np.random.randint(-170, -90, 5000)
samples_y = np.random.randint(-20, 40, 5000)

samples = np.stack((samples_x, samples_y), axis=1)

for i in range(19):
    for x in samples:
        x[0] += np.random.normal(2, 1)
        x[1] += np.random.normal(1, 1)

    weights = get_weights(samples, oly_evidences, ars_evidences, pav_evidences, asc_evidences, i)

    samples = samples[weights.argsort()]
    samples = samples[::-1]
    samples = np.delete(samples, list(range(len(samples)//2, len(samples))), axis=0)

    new_samples = []
    for j in range(len(samples)):
        new_samples.append([samples[j, 0] + np.random.rand(), samples[j, 1] + np.random.rand()])

    new_samples = np.array(new_samples)
    samples = np.concatenate((samples, new_samples))

weights = get_weights(samples, oly_evidences, ars_evidences, pav_evidences, asc_evidences, 19)

result = np.average(samples, axis=0, weights=weights)

x = int(np.ceil((result[0]/10)) * 10)

y = int(np.ceil((result[1]/10)) * 10)

print(x)
print(y)



