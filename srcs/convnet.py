import sys, os
sys.path.append(os.pardir)
import pickle
import numpy as np
from collections import OrderedDict
from srcs.layers import *
from srcs.functions import numerical_gradient


class ConvNet:
    def __init__(self, input_dim=(1, 28, 28),
                 conv_param={'filter_num': 30, 'filter_size': 5, 'pad': 0, 'stride': 1},
                 hidden_size=100, output_size=48, weight_init_std=0.01):
        filter_num = conv_param['filter_num']
        filter_size = conv_param['filter_size']
        filter_pad = conv_param['pad']
        filter_stride = conv_param['stride']
        input_size = input_dim[1]
        conv_output_size = (input_size - filter_size + 2 * filter_pad) / filter_stride + 1
        conv_output_size = (conv_output_size / 2 - filter_size + 2 * filter_pad) / filter_stride + 1
        pool_output_size = int(filter_num * (conv_output_size / 2) * (conv_output_size / 2))

        # 重みの初期化
        self.params = {}
        self.params['W1'] = weight_init_std * \
                            np.random.randn(filter_num, input_dim[0], filter_size, filter_size)
        self.params['b1'] = np.zeros(filter_num)
        self.params['W2'] = weight_init_std * \
                            np.random.randn(filter_num, filter_num, filter_size, filter_size)
        self.params['b2'] = np.zeros(filter_num)
        self.params['W3'] = weight_init_std * \
                            np.random.randn(pool_output_size, hidden_size)
        self.params['b3'] = np.zeros(hidden_size)
        self.params['W4'] = weight_init_std * \
                            np.random.randn(hidden_size, output_size)
        self.params['b4'] = np.zeros(output_size)

        # レイヤの生成
        self.layers = OrderedDict()
        self.layers['Conv1'] = Convolution(self.params['W1'], self.params['b1'],
                                           conv_param['stride'], conv_param['pad'])
        self.layers['Relu1'] = Relu()
        self.layers['Pool1'] = Pooling(pool_h=2, pool_w=2, stride=2)
        self.layers['Conv2'] = Convolution(self.params['W2'], self.params['b2'],
                                           conv_param['stride'], conv_param['pad'])
        self.layers['Relu2'] = Relu()
        self.layers['Pool2'] = Pooling(pool_h=2, pool_w=2, stride=2)
        self.layers['Affine1'] = Affine(self.params['W3'], self.params['b3'])
        self.layers['Relu3'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W4'], self.params['b4'])

        self.last_layer = SoftmaxWithLoss()

    def predict(self, x):
        layer_values = self.layers.values()
        for layer in layer_values:
            x = layer.forward(x)
        return x

    def predict_3_char(self, x):
        x = self.predict(x)

        # 前後の文字の連続する度合で補正
        code_reli_file = "code_reliability.npy"
        if os.path.exists(code_reli_file):
            code_reli = np.load(code_reli_file)
            code_reli_t = code_reli.T
            set_num = int(x.shape[0] / 3)
            x = np.reshape(x, (set_num, 3, 48))

            max = np.amax(x, axis=2, keepdims=True)  # 各1文字で最大の値
            max_idx = np.argmax(x, axis=2)  # その文字種 0-47
            max_num_idx = np.argmax(max, axis=1)
            max_num_idx = np.reshape(max_num_idx, (-1))  # 3文字の中で最大の値をもつ文字位置 0-2
            max_char = np.diag(max_idx[:, max_num_idx])  # その文字種 0-47

            for i in range(set_num):
                # print("bf: [{}, {}, {}]".format(np.argmax(x[i][0]), np.argmax(x[i][1]), np.argmax(x[i][2])))  # for debug
                base_idx = max_num_idx[i]
                base_char = max_char[i]
                if base_idx == 0:
                    x[i][1] += code_reli[base_char]
                    x[i][2] += code_reli[np.argmax(x[i][1])]
                elif base_idx == 1:
                    x[i][0] += code_reli_t[base_char]
                    x[i][2] += code_reli[base_char]
                else:
                    x[i][1] += code_reli_t[base_char]
                    x[i][0] += code_reli_t[np.argmax(x[i][1])]
                # print("af: [{}, {}, {}]".format(np.argmax(x[i][0]), np.argmax(x[i][1]), np.argmax(x[i][2])))  # for debug

            x = np.reshape(x, (set_num * 3, 48))

        else:
            print("error: not found file in ConvNet.predict_3_char")

        return x

    def loss(self, x, t):
        y = self.predict(x)
        return self.last_layer.forward(y, t)

    def accuracy(self, x, t, batch_size=120):
        if t.ndim != 1:
            t = np.argmax(t, axis=1)

        acc_list = np.empty(0)
        acc = 0.0
        char_3_acc = 0.0

        x_len = x.shape[0]
        y = np.empty((0, 48))
        for i in range(0, x_len, batch_size):
            pred = self.predict_3_char(x[i:(i + batch_size)])
            y = np.append(y, pred, axis=0)
        y = np.argmax(y, axis=1)
        acc_list = np.append(acc_list, (y == t))

        acc = np.sum(acc_list) / x.shape[0]
        char_3_acc = np.floor(acc_list.reshape(-1, 3).sum(axis=1) / 3).sum() * 3 / x.shape[0]

        return acc, char_3_acc

    def numerical_gradient(self, x, t):
        loss_w = lambda w: self.loss(x, t)

        grads = {}
        for idx in (1, 2, 3, 4):
            grads['W' + str(idx)] = numerical_gradient(loss_w, self.params['W' + str(idx)])
            grads['b' + str(idx)] = numerical_gradient(loss_w, self.params['b' + str(idx)])

        return grads

    def gradient(self, x, t):
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 設定
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Conv1'].dW, self.layers['Conv1'].db
        grads['W2'], grads['b2'] = self.layers['Conv2'].dW, self.layers['Conv2'].db
        grads['W3'], grads['b3'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W4'], grads['b4'] = self.layers['Affine2'].dW, self.layers['Affine2'].db

        return grads

    def save_params(self, file_name="params.pkl"):
        params = {}
        for key, val in self.params.items():
            params[key] = val
        with open(file_name, 'wb') as f:
            pickle.dump(params, f)

    def load_params(self, file_name="params.pkl"):
        with open(file_name, 'rb') as f:
            params = pickle.load(f)
        for key, val in params.items():
            self.params[key] = val

        for i, key in enumerate(['Conv1', 'Conv2', 'Affine1', 'Affine2']):
            self.layers[key].W = self.params['W' + str(i + 1)]
            self.layers[key].b = self.params['b' + str(i + 1)]
