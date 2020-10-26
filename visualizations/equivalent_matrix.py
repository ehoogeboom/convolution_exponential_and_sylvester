import torch
import models.transformations.convexp as convexp
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

target = [83., 124., 181.]

target_orange = [255., 186., 126.]

# top = cm.get_cmap('Oranges_r', 128)
# bottom = cm.get_cmap('Blues', 128)
#
# newcolors = np.vstack((top(np.linspace(0, 1, 128)),
#                        bottom(np.linspace(0, 1, 128))))
# newcmp = ListedColormap(newcolors, name='OrangeBlue')

N = 256
blues = np.ones((N, 4))
blues[:, 0] = np.linspace(83./256, 1, N)
blues[:, 1] = np.linspace(124./256, 1, N)
blues[:, 2] = np.linspace(181./256, 1, N)
blues = blues[::-1]
newcmp = ListedColormap(blues)
oranges = np.ones((N, 4))
oranges[:, 0] = np.linspace(target_orange[0] / 256, 1, N)
oranges[:, 1] = np.linspace(target_orange[1] / 256, 1, N)
oranges[:, 2] = np.linspace(target_orange[2] / 256, 1, N)

target_green = [105, 122, 90]
greens = np.ones((N, 4))
greens[:, 0] = np.linspace(target_green[0] / 256, 1, N)
greens[:, 1] = np.linspace(target_green[1] / 256, 1, N)
greens[:, 2] = np.linspace(target_green[2] / 256, 1, N)
greens = greens[::-1]
green_cmp = ListedColormap(greens)

cmp_orange_blue = ListedColormap(np.vstack((oranges, blues)), name='OrangeBlue')


def save_matrix(M, name):
    plt.imshow(M, cmap=newcmp, vmin=0, vmax=1.2)
    plt.axis('off')
    plt.savefig(name + ".svg", bbox_inches='tight')


def matrix_exp(M, terms=10, plot_matrices=False, path=''):
    d = M.size(0)
    assert d == M.size(1)

    prod = torch.eye(d)

    if plot_matrices:
        save_matrix(prod, path + 'matrix_term_{}'.format(0))
    result = prod
    for i in range(1, terms+1):
        prod = torch.mm(prod, M) / i
        if plot_matrices and i < 5:
            save_matrix(prod, path + 'matrix_term_{}'.format(i))
        result = result + prod

    return result


def to_matrix_index(c_idx, C, h_idx, H, w_idx, W):
    return c_idx * (H * W) + h_idx * W + w_idx


def create_equivalent_matrix(kernel, input_size):
    path = 'equiv_matrix/'
    os.makedirs(path, exist_ok=True)

    def convert_kernel_location(y, x, j, i, K1, K2):
        m1 = (K1 - 1) // 2
        m2 = (K2 - 1) // 2
        return y + j - m1, x + i - m2

    C_out, C_in, K1, K2 = kernel.size()

    assert C_in == input_size[0]

    H, W = input_size[1:]

    equiv_matrix = torch.zeros((C_out * H * W, C_in * H * W))

    inp = torch.randn((C_in, H, W))
    out_manual_conv = torch.zeros((C_out, H, W))

    for cout in range(C_out):
        for y in range(H):
            for x in range(W):
                for cin in range(C_in):
                    for j in range(K1):
                        for i in range(K2):
                            v, u = convert_kernel_location(y, x, j, i, K1, K2)

                            if v < 0 or v >= H or u < 0 or u >= W:
                                continue

                            out_manual_conv[cout, y, x] += kernel[cout, cin, j, i] * inp[cin, v, u]

                            j_matrix = to_matrix_index(cout, C_out, y, H, x, W)
                            i_matrix = to_matrix_index(cin, C_in, v, H, u, W)
                            assert equiv_matrix[j_matrix, i_matrix] == 0.
                            equiv_matrix[j_matrix, i_matrix] = kernel[cout, cin, j, i]

    out_pytorch_conv = F.conv2d(inp[None], weight=kernel, bias=None, padding=1)

    inp_vector = inp.view(-1, 1)
    out_vector = torch.mm(equiv_matrix, inp_vector).view(C_out, H, W)

    print('matrixmul vs pytorch',
          torch.mean(torch.abs(out_vector - out_pytorch_conv)))

    print('manual vs pytorch',
          torch.mean(torch.abs(out_manual_conv - out_pytorch_conv)))

    equiv_matrix_exp = matrix_exp(equiv_matrix, terms=15, plot_matrices=True, path=path)
    inv_equiv_matrix_exp = matrix_exp(-equiv_matrix, terms=15)
    out_vector_exp = torch.mm(equiv_matrix_exp, inp_vector).view(C_out, H, W)

    out_pytorch_expconv = convexp.functional.conv_exp(inp[None], kernel, terms=15)

    print('exp: matrix vs pytorch',
          torch.mean(torch.abs(out_vector_exp - out_pytorch_expconv)))

    plt.imshow(equiv_matrix, cmap=newcmp, vmin=0., vmax=1.2)
    plt.axis('off')
    plt.savefig(path + "equiv_matrix.svg", bbox_inches='tight')

    plt.imshow(equiv_matrix_exp, cmap=newcmp, vmin=0.,  vmax=1.2)
    plt.axis('off')
    plt.savefig(path + "equiv_matrix_exp.svg", bbox_inches='tight')

    plt.imshow(inv_equiv_matrix_exp, cmap=newcmp, vmin=0.,  vmax=1.2)
    plt.axis('off')
    plt.savefig(path + "inv_equiv_matrix_exp.svg", bbox_inches='tight')

    for cin in range(C_in):
        for cout in range(C_out):
            plt.imshow(
                kernel[cout, cin], cmap=newcmp, vmin=0., vmax=1.2)
            plt.axis('off')
            plt.savefig(path + "kernel_out_{}_in_{}.svg".format(cout, cin),
                        bbox_inches='tight')


def show_resulting_feature_maps(input, kernel, name_prefix):
    # N = 256
    # vals = np.ones((N, 4))
    # vals[:, 0] = np.linspace(83. / 256, 1, N)
    # vals[:, 1] = np.linspace(124. / 256, 1, N)
    # vals[:, 2] = np.linspace(181. / 256, 1, N)
    # vals = vals[::-1]
    # newcmp = ListedColormap(vals)


    out_expconv = convexp.functional.conv_exp(
        input, kernel, terms=15)
    print(out_expconv.max())

    # Add first term.
    results_ith_term = [input]

    print(results_ith_term[0].max(), 'term', 0)

    tmp = F.conv2d(input, kernel, padding=(1, 1))
    print(kernel)
    print(tmp.max(), 'tmpmax')

    for i in range(1, 5):
        out_expconv_i_terms = (
            convexp.functional.conv_exp(input, kernel, terms=i) -
            convexp.functional.conv_exp(input, kernel, terms=i-1)
        )

        print(out_expconv_i_terms.max(), 'term', i)

        results_ith_term += [out_expconv_i_terms]

    for i, ith_term in enumerate(results_ith_term):
        print('{}th term'.format(i))
        plt.imshow(ith_term.squeeze(), cmap=cmp_orange_blue, vmin=-1.3, vmax=1.3)
        plt.axis('off')
        plt.savefig(name_prefix + "_term_{}.svg".format(i), bbox_inches='tight')

    print('convexp'.format(i))
    plt.imshow(out_expconv.squeeze(), cmap=cmp_orange_blue, vmin=-1.3, vmax=1.3)
    plt.axis('off')
    plt.colorbar()
    plt.savefig(name_prefix + "_total.svg".format(i), bbox_inches='tight')
    plt.close()

    return out_expconv

def plot_image_as_vector():
    path = 'equiv_matrix/'
    os.makedirs(path, exist_ok=True)

    image = np.linspace(0.1, .6, 5 * 5).reshape(5, 5)

    plt.imshow(image, cmap=green_cmp, vmin=0., vmax=1.)
    plt.axis('off')
    plt.savefig(path + "image.svg", bbox_inches='tight')
    image = image.reshape(5*5, 1)

    plt.imshow(image, cmap=green_cmp, vmin=0., vmax=1.)
    plt.savefig(path + "image_vector.svg", bbox_inches='tight', vmin=0.)
    plt.axis('off')


if __name__ == '__main__':
    handcrafted = torch.Tensor(
        [[.2, .4, .2],
         [.4, 0., .4],
         [.2, .4, .2]]
    ).view(1, 1, 3, 3)

    handcrafted_edge = torch.Tensor(
        [[.0, .0, -.0],
         [0.6, 0., -.6],
         [.0, .0, -.0]]
    ).view(1, 1, 3, 3)

    kernel = torch.randn((1, 1, 3, 3)) / 3

    input_size = (1, 5, 5)

    plot_image_as_vector()

    # create_equivalent_matrix(handcrafted, input_size)

    # from utils.load_data import load_bmnist
    # import utils.load_data
    # utils.load_data.ROOT = '../data'
    #
    # class Args:
    #     batch_size = 1000
    # args = Args()
    # _, _, test_loader, _ = load_bmnist(args)
    #
    # x_example = None
    # for idx, (x, _) in enumerate(test_loader):
    #     if idx == 8:
    #         x_example = x[500][None]
    #         break
    #
    # os.makedirs('convexp_feature_maps/', exist_ok=True)
    #
    # out = show_resulting_feature_maps(
    #     x_example, handcrafted_edge,
    #     name_prefix='convexp_feature_maps/convexp_fm')
    # recon = show_resulting_feature_maps(
    #     out, -handcrafted_edge,
    #     name_prefix='convexp_feature_maps/inv_convexp_fm')
