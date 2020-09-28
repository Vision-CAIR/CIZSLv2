import matplotlib.pyplot as plt
import os
import numpy as np


# mfile_path = 'process_100.txt'
# save_image_path = 'test.png'
noloss_path = 'noloss-all.txt'
loss_path = 'loss-all.txt'
save_image_path = 'discriminative_cub_easy_last1000.png'

with open(noloss_path, 'r') as noloss, open(loss_path, 'r') as loss:
    ite = 0
    y_axis, z_axis = [], []
    for line in noloss:
        if (ite > 10000) and (ite < 14800):
            y_axis.append(float(line))
        ite += 1
    ite = 0
    for line in loss:
        if (ite > 10000) and (ite < 14800):
            z_axis.append(float(line))
        ite += 1

    # print(len(y_axis))
    x_axis = np.arange(max(len(y_axis), len(z_axis)))

    plt.title('Highest classification confidence with last 1k batches')
    plt.plot(x_axis, y_axis, color='blue', label='w/o discriminative loss', linestyle='-')
    plt.plot(x_axis, z_axis, color='green', label='w discriminative loss', linestyle='-')
    plt.legend()
    plt.savefig(save_image_path)
    plt.show()

