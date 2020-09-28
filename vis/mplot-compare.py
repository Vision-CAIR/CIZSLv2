import matplotlib.pyplot as plt
import os
import numpy as np


# mfile_path = 'process_100.txt'
# save_image_path = 'test.png'
noloss_path = 'noloss-all.txt'
loss_path = 'loss-all.txt'
# noloss_path = 'constrains_noloss_cub1_easy1.txt'
# loss_path = 'constrains_loss_cub1_easy1.txt'
save_image_path = 'discriminative_cub_easy2.png'


with open(noloss_path, 'r') as noloss, open(loss_path, 'r') as loss:
    ite = 0
    y_axis, z_axis = [], []
    for yline in noloss:
        if ite > 14950:
            continue
        ite += 1
        y_axis.append(float(yline))
    ite = 0
    for zline in loss:
        if ite > 14950:
            continue
        ite += 1
        z_axis.append(float(zline))
    # print(len(y_axis))
    x_axis = np.arange(max(len(y_axis), len(z_axis)))

    plt.title('Highest classification confidence with all batches')
    plt.plot(x_axis, y_axis, color='blue', label='w/o discriminative loss', linestyle='-')
    plt.plot(x_axis, z_axis, color='green', label='w discriminative loss', linestyle='-')
    plt.legend()
    plt.savefig(save_image_path)
    plt.show()

