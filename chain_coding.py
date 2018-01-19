from PIL import Image
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import cv2
import scipy.io as sio
from transforms import image_resize


def chain_code(img):

    img = image_resize(img,(28,28),reverse=True)
    # img = np.reshape(img, (-1, 28)).astype(np.uint8)
    # plt.imshow(img, cmap='Greys')
    # plt.show()

    ## Discover the first point
    start_point=(0,0)
    for i, row in enumerate(img):
        for j, value in enumerate(row):
            if value == 255:
                start_point = (i, j)
                print(start_point, value)
                break
        else:
            continue
        break
    plt.imshow(img, cmap='Greys')
    plt.pause(0.5)
    print(start_point,value)
    plt.show()
    directions = [0, 1, 2,
                  7, 3,
                  6, 5, 4]
    dir2idx = dict(zip(directions, range(len(directions))))

    change_j = [-1, 0, 1,  # x or columns
                -1, 1,
                -1, 0, 1]

    change_i = [-1, -1, -1,  # y or rows
                0, 0,
                1, 1, 1]

    border = []
    chain = []
    curr_point = start_point
    for direction in directions:
        idx = dir2idx[direction]
        new_point = (start_point[0] + change_i[idx], start_point[1] + change_j[idx])
        if img[new_point] != 0:  # if is ROI
            border.append(new_point)
            chain.append(direction)
            curr_point = new_point
            break
    count = 0
    while curr_point != start_point:
        # figure direction to start search
        b_direction = (direction + 5) % 8
        dirs_1 = range(b_direction, 8)
        dirs_2 = range(0, b_direction)
        dirs = []
        dirs.extend(dirs_1)
        dirs.extend(dirs_2)
        for direction in dirs:
            idx = dir2idx[direction]
            new_point = (curr_point[0] + change_i[idx], curr_point[1] + change_j[idx])
            if img[new_point] != 0:  # if is ROI
                border.append(new_point)
                chain.append(direction)
                curr_point = new_point
                break
        if count == 1000: break
        count += 1
    print(count)
    print(chain)
    # plt.imshow(img, cmap='Greys')
    plt.plot([i[1] for i in border], [i[0] for i in border])
    plt.pause(4)


mat_contents = sio.loadmat("nisttrain_cell/file_0016.mat")
for image in mat_contents['imcells'][0]:
    chain_code(image)
plt.show()
# img = Image.open("testimage2.jpg")
# iar = np.asarray(img)
# chain_code(iar)
# plt.show()