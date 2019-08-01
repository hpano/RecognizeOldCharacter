import os
from numba import jit
import numpy as np
import pandas as pd
from pathlib import Path
from srcs.utils import AlconDataset, AlconTargets, unicode_to_kana_list, evaluation, code2index
from PIL import Image


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
        # print(d_sum)
        tf = code_reliability / d_sum
        # print(tf)
        td_num = np.count_nonzero(code_reliability > 0, axis=0) + 1
        # print(td_num)
        idf = np.log2(48 / td_num)
        # print(idf)
        tfidf = tf * idf
        # print(tfidf)
        # 何らかの補正
        code_reliability = tfidf + 0

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
