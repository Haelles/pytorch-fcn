import os
import os.path as osp
from shutil import copyfile


def gen():
    lines = open('./torchfcn/datasets/dp_train_val.txt').readlines()
    root = '../fashion_upload/datasets/dp'
    out = './dp'
    if not osp.exists(out):
        os.makedirs(out)
    for l in lines:
        sub_path, tag = l.split()[0].replace('.jpg', '_gray.png'), l.split()[-1]
        pos = sub_path.rfind('/')
        # img = 'img_320_512_image'
        parsing = 'img_320_512_parsing'
        target_file = osp.join(out, parsing, sub_path)
        if not osp.exists(osp.join(out, parsing, sub_path[0: pos])):
            os.makedirs(osp.join(out, parsing, sub_path[0: pos]))
        source_file = osp.join(root, 'img_320_512_parsing', sub_path)
        copyfile(source_file, target_file)


if __name__ == '__main__':
    gen()
