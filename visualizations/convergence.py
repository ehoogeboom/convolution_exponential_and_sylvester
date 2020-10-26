
import numpy as np
import scipy as sp
import scipy.special
import matplotlib.pyplot as plt


blue = np.array([83., 124., 181.]) / 255.

blue_strong = np.array([98, 147, 217]) / 255.

# orange = np.array([255., 186., 126.]) / 255.

orange_strong = np.array([255, 168, 92]) / 255.

green = np.array([105., 122., 90.]) / 255.

green_strong = np.array([149, 173, 127.]) / 255.


if __name__ == '__main__':
    plt.figure(figsize=(5, 3))
    # model = torch.load()
    xrange = 10

    x = np.linspace(0, 10, 300)
    x_integers = np.arange(0, 10+1, 1)

    gamma = sp.special.gamma(x + 1)
    factorial = sp.special.gamma(x_integers + 1)

    markers = ['o', 'x', 's']
    colors = [blue_strong, orange_strong, green_strong]

    cs = [0.5, 1., 2.]

    for i, c in enumerate(reversed(cs)):
        y = np.power(c, x) / gamma

        y_ints = np.power(c, x_integers) / factorial

        # plt.semilogy(x, y, label='Lip(W) = {}'.format(c, y_max))
        plt.plot(x, y, color=colors[i])
        plt.plot(
            x_integers, y_ints, marker=markers[i], linewidth=0.,
            label='||M||_p = {}'.format(c), color=colors[i])

    plt.xlabel(r'iteration i', size=10)
    # plt.ylabel(r'$\frac{c^i}{i!}$', rotation=0, size=14)
    plt.ylabel(r'$inf_x \frac{||M^i x||_p}{||x||_p i!}$', rotation=0, size=16)

    ticks = np.arange(0, xrange + 1, 1)
    plt.xticks(ticks, [str(t) if t % 2 == 0 else '' for t in ticks])
    # plt.ylim(0, 10)
    plt.legend()
    plt.savefig('./convergence.svg')
