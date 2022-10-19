import matplotlib.pyplot as plt
import numpy as np

def draw_points(func, x_1_sequence, x_2_sequence):
    fig = plt.figure(figsize=plt.figaspect(0.5))
    X1, X2 = np.meshgrid(np.linspace(-15.0, 15.0, 1000), np.linspace(-15.0, 15.0, 1000))
    Y = func(X1, X2)
    f_sequence = [func(x_1_sequence[i], x_2_sequence[i]) for i in range(len(x_1_sequence))]

    # First subplot
    ax = fig.add_subplot(1, 2, 1)

    cp = ax.contour(X1, X2, Y, colors='black', linestyles='dashed', linewidths=1)
    ax.clabel(cp, inline=1, fontsize=10)
    cp = ax.contourf(X1, X2, Y, )
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.scatter(x_1_sequence, x_2_sequence, s=10, c="y")

    # Second subplot
    ax = fig.add_subplot(1, 2, 2, projection='3d')

    ax.contour3D(X1, X2, Y, 50, cmap="Blues")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.scatter3D(x_1_sequence, x_2_sequence, f_sequence, s=10, c="r")

    plt.show()