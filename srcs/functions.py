import os
from numba import jit
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image


def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T

    x = x - np.max(x)  # オーバーフロー対策
    return np.exp(x) / np.sum(np.exp(x))


def cross_entropy_error(y, t):
    delta = 1e-7
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + delta)) / batch_size


def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)  # f(x + h)

        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp_val  # 値を元に戻す
        it.iternext()

    return grad


def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    """
    Parameters
    ----------
    input_data : (データ数, チャンネル, 高さ, 幅)の4次元配列からなる入力データ
    filter_h : フィルターの高さ
    filter_w : フィルターの幅
    stride : ストライド
    pad : パディング
    Returns
    -------
    col : 2次元配列
    """

    N, C, H, W = input_data.shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1

    img = np.pad(input_data, [(0, 0), (0, 0), (pad, pad), (pad, pad)], 'constant')

    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))
    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]
    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)

    return col


def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    """
    Parameters
    ----------
    col :
    input_shape : 入力データの形状（例：(10, 1, 28, 28)）
    filter_h :
    filter_w
    stride
    pad
    Returns
    -------
    """
    N, C, H, W = input_shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((N, C, H + 2 * pad + stride - 1, W + 2 * pad + stride - 1))
    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]


def index2code(index):
    codes = ["U+3042", "U+3044", "U+3046", "U+3048", "U+304A", "U+304B", "U+304D", "U+304F",
             "U+3051", "U+3053", "U+3055", "U+3057", "U+3059", "U+305B", "U+305D", "U+305F",
             "U+3061", "U+3064", "U+3066", "U+3068", "U+306A", "U+306B", "U+306C", "U+306D", "U+306E", "U+306F",
             "U+3072", "U+3075", "U+3078", "U+307B", "U+307E", "U+307F",
             "U+3080", "U+3081", "U+3082", "U+3084", "U+3086", "U+3088", "U+3089", "U+308A", "U+308B", "U+308C", "U+308D", "U+308F",
             "U+3090", "U+3091", "U+3092", "U+3093"]
    return codes[index]


def code2index(code):
    codes = ["U+3042", "U+3044", "U+3046", "U+3048", "U+304A", "U+304B", "U+304D", "U+304F",
             "U+3051", "U+3053", "U+3055", "U+3057", "U+3059", "U+305B", "U+305D", "U+305F",
             "U+3061", "U+3064", "U+3066", "U+3068", "U+306A", "U+306B", "U+306C", "U+306D", "U+306E", "U+306F",
             "U+3072", "U+3075", "U+3078", "U+307B", "U+307E", "U+307F",
             "U+3080", "U+3081", "U+3082", "U+3084", "U+3086", "U+3088", "U+3089", "U+308A", "U+308B", "U+308C", "U+308D", "U+308F",
             "U+3090", "U+3091", "U+3092", "U+3093"]
    return codes.index(code)


def set_data(targets, target_name, num_train, filter, img_size):
    load_size = min(num_train, len(targets))
    isTest = (target_name == "testdata")

    div_targets = np.empty((0, 1, img_size, img_size))
    div_codes = np.empty((0, 48), dtype=np.uint8)

    if not os.path.exists("./npy/{}_{}_img.npy".format(target_name, load_size)):
        div_targets = np.empty((0, 1, img_size, img_size))
        div_codes = np.empty((0, 48))

        msg = "loading {}".format(target_name)
        for i in range(load_size):
            print_progress_bar(msg, i, load_size)

            img, codes = targets[i]
            div_targets = np.append(div_targets, div_img(targets[i][0], filter, img_size), axis=0)
            if not isTest:
                one_hot = np.identity(48, dtype=np.uint8)[codes]
                div_codes = np.append(div_codes, one_hot, axis=0)
        print_progress_bar(msg, i + 1, load_size)

        np.save("./npy/{}_{}_img".format(target_name, load_size), div_targets)
        np.save("./npy/{}_{}_code".format(target_name, load_size), div_codes)

    else:
        div_targets = np.load("./npy/{}_{}_img.npy".format(target_name, load_size))
        div_codes = np.load("./npy/{}_{}_code.npy".format(target_name, load_size))

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


def calc_code_reliability():
    file_name = "code_reliability"
    if not os.path.exists(file_name + ".npy"):
        code_reliability = np.zeros((48, 48), int)

        fnm = Path("./dataset/train/annotations.csv")
        assert fnm.exists()
        df = pd.read_csv(fnm)
        df_len = len(df)
        for idx in range(df_len):
            msg = "loading code_reli"
            print_progress_bar(msg, idx, df_len)

            unicodes = list(df.iloc[idx, 1:4])
            code0 = code2index(unicodes[0])
            code1 = code2index(unicodes[1])
            code2 = code2index(unicodes[2])
            code_reliability[code0][code1] += 1
            code_reliability[code1][code2] += 1
        print_progress_bar(msg, idx + 1, df_len)

        # TF-IDFによる単語の重要度計算
        d_sum = np.sum(code_reliability, axis=1, keepdims=True)
        tf = code_reliability / d_sum
        td_num = np.count_nonzero(code_reliability > 0, axis=0) + 1
        idf = np.log2(48 / td_num)
        tfidf = tf * idf
        # CNNの結果と合わせるための重みづけ
        code_reliability = tfidf * 49

        np.save(file_name, code_reliability)


def print_progress_bar(msg, current_progress, total_size):
    pro_size = 20
    msg_size = 20
    pro_rate = int((current_progress / total_size) * pro_size)
    pro_bar = ("=" * (pro_rate)) + (" " * int(pro_size - pro_rate))
    msg += " " * (msg_size - len(msg))
    print("\r{}[{}] {}/{}".format(msg, pro_bar, current_progress, total_size), end="")
    if current_progress == total_size:
        print()
