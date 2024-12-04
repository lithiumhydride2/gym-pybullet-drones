import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from itertools import product
import time

fig, ax = plt.subplots()
grid = np.array(list(product(np.linspace(-1, 1, 40), np.linspace(
    -1,
    1,
    40,
))))
quad_mesh = ax.pcolormesh(grid[:, 0].reshape((40, 40)),
                          grid[:, 1].reshape((40, 40)),
                          np.random.random((40, 40)),
                          shading='auto',
                          vmin=0,
                          vmax=1)
ax: matplotlib.axes.Axes
for _ in range(100):
    quad_mesh.set_array(np.random.random((40, 40)))
    plt.show()
    time.sleep(0.1)
