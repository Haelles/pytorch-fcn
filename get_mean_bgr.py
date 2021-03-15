import numpy as np
import cv2
import os
import os.path as osp


def cal():
    means, stdevs = [], []
    img_list = []
    root = '../fashion_upload/datasets/dp'
    lines = open('./torchfcn/datasets/dp_train_val.txt').readlines()
    imgs_path_list = []
    for l in lines:
        sub_path, tag = l.split()[0], l.split()[-1]
        if tag == 'train':
            image_path = osp.join(root, 'img_320_512_image', sub_path)
            imgs_path_list.append(image_path)

    for item in imgs_path_list:
        img = cv2.imread(item)  # BGR
        img = img[:, :, :, np.newaxis]
        img_list.append(img)

    imgs = np.concatenate(img_list, axis=3)
    imgs = imgs.astype(np.float32) / 255.

    for i in range(3):
        pixels = imgs[:, :, i, :].ravel()  # 拉成一行
        means.append(np.mean(pixels))
        stdevs.append(np.std(pixels))

    # BGR --> RGB ， CV读取的需要转换，PIL读取的不用转换
    # 由于DeepFahsion.transform中为mean_bgr 因此不用转换
    # means.reverse()
    # stdevs.reverse()

    print("normMean = {}".format(means))
    print("normStd = {}".format(stdevs))
    return means, stdevs

    # 包括测试集
    # normMean = [0.7795435, 0.7467555, 0.7355003]
    # normStd = [0.2669851, 0.283014, 0.2905218]


if __name__ == '__main__':
    normMean, normStd = cal()
    # train dataset
    # normMean = [0.73515534, 0.7464929, 0.7790557]
    # normStd = [0.28991002, 0.28232634, 0.2663163]
    # [187.4646117 190.3556895 198.6592035]
    # [73.9270551 71.9932167 67.9106565]
    print(np.array(normMean) * 255)
    print(np.array(normStd) * 255)
