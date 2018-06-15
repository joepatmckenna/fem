import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(1, 2, figsize=(12, 3))

s = [3 * i + np.random.randint(3) for i in range(2)]

ix = [0, 1, 2, 6, 7, 8]
iy = [2, 3]

xx = np.linspace(0, 1, 9)[ix]
xy = np.zeros(9)[ix]
yx = np.linspace(0, 1, 6)[iy]
yy = np.ones(6)[iy]

for a in ax:
    a.scatter(yx, yy, c='w', edgecolor='k', s=100, clip_on=False)
    for i in range(2):
        a.text(
            yx[i],
            1.25,
            '$h_%i$' % (i + 1, ),
            va='center',
            ha='center',
            fontsize=10)

ax[0].scatter(xx, xy, c='w', edgecolor='k', s=100, clip_on=False)
for i in range(6):
    ax[0].text(
        xx[i],
        -0.25,
        '$\sigma_{%i%i}$' % (i / 3 + 1, i % 3 + 1),
        va='center',
        ha='center',
        fontsize=10)
for i in range(2):
    ax[0].text(
        xx[3 * i + 1],
        -0.5,
        '$x_%s$' % (i + 1, ),
        va='center',
        ha='center',
        fontsize=10)

for i in range(2):
    ax[0].scatter(
        xx[s[i]], xy[s[i]], c='r', edgecolor='k', s=100, clip_on=False)
    ha = ['right', 'left']
    for j in range(2):
        x, y = xx[s[i]], xy[s[i]]
        dx, dy = yx[j] - x, yy[j] - y
        ax[0].arrow(x, y, dx, dy)
        ax[0].text(
            x + 0.5 * dx,
            y + 0.5 * dy,
            '$W^{1%i}_{%i%i}$' % (s[i] % 3 + 1, i + 1, j + 1),
            ha=ha[j],
            va='top',
            color='r')

x = np.linspace(0, 1, 9)
y = np.zeros(9)
ax[1].scatter(x, y, c='w', edgecolor='k', s=100, clip_on=False)
idx1 = ['11', '11', '11', '12', '12', '12', '13', '13', '13']
idx2 = ['21', '22', '23', '21', '22', '23', '21', '22', '23']

for i in range(9):
    ax[1].text(
        x[i],
        -.25,
        '$\sigma_{%s}\sigma_{%s}$' % (idx1[i], idx2[i]),
        va='center',
        ha='center',
        fontsize=10)
ax[1].text(x[4], -.5, '$x_1x_2$', va='center', ha='center', fontsize=10)

i = s[0] + 3 * (s[1] % 3)
ax[1].scatter(x[i], y[i], c='r', edgecolor='k', s=100)
ha = ['right', 'left']
for j in range(2):
    dx, dy = yx[j] - x[i], yy[j] - y[i]
    ax[1].arrow(x[i], y[i], dx, dy)
    ax[1].text(
        x[i] + .5 * dx,
        y[i] + .5 * dy,
        '$W^{2,(%i,%i)}_{%i,(1,2)}$' % (s[1] % 3 + 1, s[0] + 1, j + 1),
        fontsize=10,
        ha=ha[j],
        va='top',
        color='r')

for a in ax:
    a.set_ylim(-.5, 1.5)
    a.axis('off')

ax[0].set_title('example $W^1$ terms')
ax[1].set_title('example $W^2$ terms')

plt.tight_layout()
plt.show()
