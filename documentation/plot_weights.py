import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys  # only needed to determine Python version number
import os

from scipy.stats import norm

######## plot to show that weights are normally distributed
gfx_dir = 'gfx/'
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

df = pd.read_csv('weights.csv')
hist = df.hist(bins=40, edgecolor='black', linewidth=0.5)
x_axis = np.arange(-0.6, 0.6, 0.01)

if not os.path.exists(gfx_dir):
    os.mkdir(gfx_dir)

plt.plot(x_axis, hist[0][0].get_ylim()[1] * 0.285 * norm.pdf(x_axis, df.mean(axis=0)[0], np.sqrt(df.var(axis=0)[0])),
         '--')
plt.title(
    'Distribution of Weights with: ' + r'$\mu={:.4f}, \sigma^2={:.4f}$'.format(df.mean(axis=0)[0], df.var(axis=0)[0]),
    fontsize=16)
plt.xlabel('Value')
plt.ylabel('Occurrence')
plt.legend(['theoretical distribution', 'trained weights'])
plt.savefig(gfx_dir + 'weights_distribution.pdf')
plt.show()

######## plot to show that log steps are more accurate than lin steps
bits = 4
x = np.arange(-1, 1, 0.01)
y = norm.pdf(x, 0, 0.25) * 0.25

plt.figure(figsize=(5.5, 6.5))
fig = plt.subplot(2, 1, 1)
x_quant = np.linspace(-1, 1 - 2 ** (-(bits - 1)), 2 ** bits)
plt.plot(x, y, ':')
plt.fill_between(x, 0, y, alpha=0.5)
for xcoord in x_quant:
    plt.axvline(xcoord, ymin=0.05, linestyle=':', color='k', linewidth=1)

fig.set_title(
    r"4 Bit Linear Quantization for $W\sim\mathcal{N}(\mu=0,\sigma^2=0.0625)$",
    fontsize=16)
plt.xlabel('Value')
plt.ylabel('Probability')
plt.legend(['normal distribution', 'quantization'], loc=1)

fig = plt.subplot(2, 1, 2)
exp = -np.arange(2 ** (bits - 1), dtype=np.float32)
x_quant = np.concatenate([np.power(2, exp), -np.power(2, exp)])
plt.plot(x, y, ':')
plt.fill_between(x, 0, y, alpha=0.5)
for xcoord in x_quant:
    plt.axvline(xcoord, ymin=0.05, linestyle=':', color='k', linewidth=1)
fig.set_title(
    r"4 Bit $log_2$ Quantization for $W\sim\mathcal{N}(\mu=0,\sigma^2=0.0625)$",
    fontsize=16)
plt.xlabel('Value')
plt.ylabel('Probability')
plt.legend(['normal distribution', 'quantization'], loc=1)
plt.tight_layout()
plt.savefig(gfx_dir + 'lin_log_quantization.pdf')
plt.show()

######### plots to show how my binarization scheme works

M = 8
N = 5*5*32
#x = np.array([0.36896316, -0.11234736, -0.35324181, -0.06287433, -0.14378786,
#              0.34984797, 0.2745331, 0.08769812, 0.15606849, -0.40081012])
x = np.random.randn(N)
W = x

B = np.ndarray([N, M])
B_old = np.ones_like(W)
W_new = np.ndarray([N, M + 1])
W_new[:, 0] = np.copy(W)
m = 0
for i in range(M):
    B[:, i] = ((W_new[:, i] >= 0).astype(float) * 2 - 1) * B_old
    W_new[:, i + 1] = np.abs(W_new[:, i])
    B_old = B[:, i]
    # a_new = np.linalg.lstsq(B[:, range(i + 1)], W, rcond=None)[0]
    W_new[:, i + 1] = W_new[:, i + 1] - np.mean(W_new[:, i + 1])

a_new = np.linalg.lstsq(B, W, rcond=None)[0]

W_est = np.zeros_like(W)
MSE = 0
for i in range(M):
    W_est = W_est + a_new[i] * B[:, i]

MSE = np.sum(np.power(W_est - W, 2))

for m in range(M + 1):
    fig = plt.figure(m, figsize=[5, 2])
    plt.plot(W, np.zeros_like(W), 'x')
    prev_points = [0]
    dy = 0.010
    for i in range(m):
        prev_points_temp = prev_points
        for _, x_pos in enumerate(prev_points_temp):
            dx = x_pos + a_new[i]
            plt.annotate("", xy=(dx, dy), xytext=(x_pos, dy), arrowprops=dict(arrowstyle='->'))
            dx = x_pos - a_new[i]
            plt.annotate("", xy=(dx, dy), xytext=(x_pos, dy), arrowprops=dict(arrowstyle='->'))
        prev_points = prev_points + a_new[i]
        prev_points = np.concatenate([prev_points, prev_points_temp - a_new[i]])
        dy += 0.005
    plt.xlim([-np.sum(a_new) - 0.1, np.sum(a_new) + 0.1])
    plt.ylim([-0.005, 0.05])
    # Move left y-axis and bottom x-axis to centre, passing through (0,0)
    ax = plt.axes()
    # spine placement data centered
    ax.spines['left'].set_position(('data', 0.0))
    ax.spines['bottom'].set_position(('data', 0.0))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.yticks([])
    plt.xlabel('Value')
    plt.title('Definition of the {:d}. Binary Vector'.format(m))
    plt.tight_layout()
    plt.savefig(gfx_dir + 'binVec_{:d}.pdf'.format(m))
    plt.show()

#########
M = 5
x = np.arange(-1, 1, 0.01)
y = norm.pdf(x, 0, 0.25) * 0.25

u_i = np.ndarray([M])
for i in range(M):
    u_i[i] = -1 + i * 2 / (M - 1)

x_quant = 0.25 * u_i
plt.plot(x, y, ':')
plt.fill_between(x, 0, y, alpha=0.5)
for xcoord in x_quant:
    plt.axvline(xcoord, ymin=0.05, linestyle=':', color='k', linewidth=1)
plt.title(
    r"Binary Split Points for $W\sim\mathcal{N}(\mu=0,\sigma^2=0.0625)$",
    fontsize=16)
plt.xlabel('Value')
plt.ylabel('Probability')
plt.legend(['normal distribution', 'quantization'], loc=1)
plt.tight_layout()
plt.savefig(gfx_dir + 'bin_split.pdf')
plt.show()

# for i in range(N):
#     plt.figure()
#     plt.plot(W, np.zeros_like(W), 'x')
#     x = 0
#     dy = 0.005
#     for j in range(M):
#         dx = a_new[j] * B[i, j] + x
#         plt.annotate("", xy=(dx, dy), xytext=(x, dy), arrowprops=dict(arrowstyle="->"))
#         x = dx
#         dy = dy + 0.005
#     plt.plot(W[i], 0, 'or', fillstyle='none')
#     plt.show()
