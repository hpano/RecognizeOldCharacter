import numpy as np
import matplotlib.pyplot as plt
from srcs.convnet import ConvNet
from srcs.functions import index2code, set_data, calc_code_reliability, print_progress_bar
from srcs.trainer import Trainer
import time
from PIL import Image
from pathlib import Path


class MyAlgorithm():
    def __init__(self, traindata, valdata, testdata, max_epochs, num_train, batch_size, img_size, params_name):
        self.traindata = traindata
        self.valdata = valdata
        self.testdata = testdata
        self.max_epochs = max_epochs
        self.num_train = num_train
        self.batch_size = batch_size
        self.img_size = img_size
        self.params_name = params_name
        self.network = ConvNet(input_dim=(1, self.img_size, self.img_size),
                          conv_param={'filter_num': 30, 'filter_size': 5, 'pad': 0, 'stride': 1},
                          hidden_size=100, output_size=48, weight_init_std=0.01)
        filter = Image.open("filter.jpg").convert('L')
        self.filter = np.array(filter.resize((self.img_size, self.img_size)))
        calc_code_reliability()

    def build_model(self):
        # データの読み込み
        msg = "setting data"
        print_progress_bar(msg, 0, 2)
        (x_train, t_train) = set_data(self.traindata, "traindata", self.num_train, self.filter, self.img_size)
        (x_val, t_val) = set_data(self.valdata, "valdata", self.num_train, self.filter, self.img_size)
        print_progress_bar(msg, 2, 2)

        # 学習
        trainer = Trainer(self.network, x_train, t_train, x_val, t_val,
                          epochs=self.max_epochs, mini_batch_size=self.batch_size,
                          optimizer='Adam', optimizer_param={'lr': 0.001},
                          evaluate_sample_num_per_epoch=900)
        print("training")
        trainer.train()

        # パラメータの保存
        ut = int(time.time())
        params_name = "params_{}_{}_{}_{}.pkl".format(self.max_epochs, self.num_train, self.batch_size, ut)
        self.network.save_params("./params/{}".format(params_name))
        self.params_name = params_name
        print("saved network parameters as '{}'".format(params_name))

        # グラフの描画
        markers = {'train': 'o', 'test': 's'}
        x = np.arange(self.max_epochs)
        plt.plot(x, trainer.train_acc_list, marker='o', label='train', markevery=2)
        plt.plot(x, trainer.test_acc_list, marker='s', label='test', markevery=2)
        plt.plot(x, trainer.train_3_acc_list, marker='o', label='train 3', markevery=2)
        plt.plot(x, trainer.test_3_acc_list, marker='s', label='test 3', markevery=2)
        plt.xlabel("epochs")
        plt.ylabel("accuracy")
        plt.ylim(0, 1.0)
        plt.legend(loc='lower right')
        figure_name = "Figure_{}_{}_{}_{}.png".format(self.max_epochs, self.num_train, self.batch_size, ut)
        plt.savefig("./figures/{}".format(figure_name))
        print("saved figure as '{}'".format(figure_name))
        plt.show()

    def predict(self):
        sheet = self.testdata.getSheet()
        test_len = len(self.testdata)
        batch_size = 120

        # データの読み込み
        (x_test, t_test) = set_data(self.testdata, "testdata", test_len, self.filter, self.img_size)
        # # for debug
        # (x_test, t_test) = set_data(self.valdata, "valdata", self.num_train, self.filter, self.img_size)

        # 処理に時間のかかる場合はデータを削減 for debug
        test_len = 600
        test3_len = test_len * 3
        x_test, t_test = x_test[:test3_len], t_test[:test3_len]

        # パラメータの読み込み
        msg = "loading params"
        print_progress_bar(msg, 0, 1)
        param_fnm = Path("./params/{}".format(self.params_name))
        assert param_fnm.exists(), "file '{}' not exist.".format(param_fnm)
        self.network.load_params(param_fnm)
        print_progress_bar(msg, 1, 1)

        # # for debug
        # test_acc, test_3_acc = self.network.accuracy(x_test, t_test)
        # print(test_acc)
        # print(test_3_acc)

        # 文字予測
        y = np.empty((0, 48))
        msg = "predicting testdata"
        for i in range(0, test3_len, batch_size):
            print_progress_bar(msg, i, test3_len)
            pred = self.network.predict_3_char(x_test[i:(i + batch_size)])
            y = np.append(y, pred, axis=0)
        print_progress_bar(msg, i + batch_size, test3_len)
        y = np.argmax(y, axis=1)
        y = np.reshape(y, (-1, 3))

        # 文字コードに変換
        msg = "encoding testdata"
        for i in range(test_len):
            print_progress_bar(msg, i, test_len)
            sheet.iloc[i, 1:4] = [index2code(y[i][0]), index2code(y[i][1]), index2code(y[i][2])]
        print_progress_bar(msg, i + 1, test_len)

        # save predicted results in CSV
        # Zip and submit it.
        sheet.to_csv('test_prediction.csv', index=False)
