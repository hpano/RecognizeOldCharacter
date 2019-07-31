import os
from numba import jit
import numpy as np
import pandas as pd
from pathlib import Path
from srcs.utils import AlconDataset, AlconTargets, unicode_to_kana_list, evaluation, code2index
from PIL import Image


def set_data(targets, target_name, num_train, filter, img_size):
    load_size = min(num_train, len(targets))
    pro_size = 20
    memory_size = 0

    div_targets = np.empty((0, 1, img_size, img_size))
    div_codes = np.empty((0, 48), dtype=np.uint8)

    if not os.path.exists("./{}_{}_img.npy".format(target_name, load_size)):
        # print("not found")
        div_targets = np.empty((0, 1, img_size, img_size))
        div_codes = np.empty((0, 48))
        code_reliability = np.zeros((2, 48, 48), int)
    else:
        div_targets = np.load("{}_{}_img.npy".format(target_name, load_size))
        div_codes = np.load("{}_{}_code.npy".format(target_name, load_size))
        code_reliability = np.load("{}_{}_code_reliability.npy".format(target_name, load_size))
        # code_reliability = np.zeros((2, 48, 48), int)
        # memory_size = int(div_targets.shape[0] / 3)
        # print(memory_size)

    for i in range(memory_size, load_size):
        pro_rate = int((i / load_size) * pro_size)
        pro_bar = ("=" * (pro_rate)) + (" " * int(pro_size - pro_rate))
        print("\rloading {} ... [{}] {}/{}".format(target_name, pro_bar, i, load_size), end="")

        img, codes = targets[i]
        div_targets = np.append(div_targets, div_img(targets[i][0], filter, img_size), axis=0)
        one_hot = np.identity(48, dtype=np.uint8)[codes]
        div_codes = np.append(div_codes, one_hot, axis=0)

        # 要修正
        code0 = codes[0]
        code1 = codes[1]
        code2 = codes[2]
        code_reliability[1][code0][code1] += 1
        code_reliability[0][code1][code0] += 1
        code_reliability[1][code1][code2] += 1
        code_reliability[0][code2][code1] += 1

    np.save("{}_{}_img".format(target_name, load_size), div_targets)
    np.save("{}_{}_code".format(target_name, load_size), div_codes)
    np.save("{}_{}_code_reliability".format(target_name, load_size), code_reliability)
    print("\nload finished.")

    return div_targets[:load_size * 3], div_codes[:load_size * 3]


def div_img(img, filter, img_size):
    # 分割位置決定
    img_sum = img.sum(axis=1)
    img_sum = img_sum + np.append(0, img_sum[1:]) + np.append(img_sum[:(img_size * 3 - 1)], 0)
    band_rate = 0.375
    start = int(img_size * (1 - band_rate))
    end = int(img_size * (1 + band_rate))
    div_pos_1 = img_sum[start:end].argmax() + start
    rest = int((img_size * 3 - div_pos_1) / 2)
    band = int(band_rate * rest)
    start = div_pos_1 + rest - band
    end = div_pos_1 + rest + band
    div_pos_2 = img_sum[start:end].argmax() + start

    # 分割、リサイズ
    img1 = np.array(Image.fromarray(np.uint8(img[:div_pos_1])).resize((img_size, img_size)))
    img2 = np.array(Image.fromarray(np.uint8(img[div_pos_1:div_pos_2])).resize((img_size, img_size)))
    img3 = np.array(Image.fromarray(np.uint8(img[div_pos_2:])).resize((img_size, img_size)))
    img = np.array([[img1], [img2], [img3]])

    # フィルターを通す
    img = (img / 2 + filter / 2)
    img = (img > 70) * 255

    return img


def calc_code_reliability(datapath):
    if not os.path.exists("./code_reliability_test.npy"):
        code_reliability = np.zeros((48, 48), int)
        pro_size = 20

        fnm = Path(datapath) / Path('train') / 'annotations.csv'
        assert fnm.exists()
        df = pd.read_csv(fnm)
        df_len = len(df)
        for idx in range(df_len):
            pro_rate = int((idx / df_len) * pro_size)
            pro_bar = ("=" * (pro_rate)) + (" " * int(pro_size - pro_rate))
            print("\rloading code_reliability ... [{}] {}/{}".format(pro_bar, idx, df_len), end="")

            unicodes = list(df.iloc[idx, 1:4])
            code0 = code2index(unicodes[0])
            code1 = code2index(unicodes[1])
            code2 = code2index(unicodes[2])
            code_reliability[code0][code1] += 1
            code_reliability[code1][code2] += 1

        np.save("code_reliability_test", code_reliability)
        print()

    else:
        code_reliability = np.load("code_reliability_test.npy")
        print(code_reliability)

        # TF-IDFによる単語の重要度計算
        d_sum = np.sum(code_reliability, axis=1, keepdims=True)
        print(d_sum)
        tf = code_reliability / d_sum
        print(tf)
        td_num = np.count_nonzero(code_reliability > 0, axis=0) + 1
        print(td_num)
        idf = np.log(48 / td_num)
        print(idf)

    return code_reliability


# Set dataset path manually
datapath = './dataset/'

# Load dataset
# targets = AlconTargets(datapath, train_ratio=0.99)
# traindata = AlconDataset(datapath, targets.train, isTrainVal=True)
# num_train = 12000
# img_size = 32
# filter = Image.open("filter2.jpg").convert('L')
# filter = np.array(filter.resize((img_size, img_size)))
np.set_printoptions(threshold=np.inf)

# (x_train, t_train) = set_data(traindata, "test", num_train, filter, img_size)
# print(t_train)  # for debug
# for i in range(num_train * 3):
#     imgg = Image.fromarray(np.uint8(x_train[i][0]))
#     imgg.show()
# code_reliability = np.load("code_reliability_test.npy")
code_reliability = calc_code_reliability(datapath)
# print(code_reliability)
