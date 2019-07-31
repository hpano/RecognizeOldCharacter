import numpy as np
import matplotlib.pyplot as plt
from srcs.convnet import ConvNet
from srcs.functions import set_data
from srcs.trainer import Trainer
import time
from PIL import Image


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

    def build_model(self):
        # データの読み込み
        print("setting data ...")
        # code_reliability
        (x_train, t_train) = set_data(self.traindata, "traindata", self.num_train, self.filter, self.img_size)
        (x_val, t_val) = set_data(self.valdata, "valdata", self.num_train, self.filter, self.img_size)

        # 学習
        trainer = Trainer(self.network, x_train, t_train, x_val, t_val,
                          epochs=self.max_epochs, mini_batch_size=self.batch_size,
                          optimizer='Adam', optimizer_param={'lr': 0.001},
                          evaluate_sample_num_per_epoch=900)
        print("train ...")
        trainer.train()

        # パラメータの保存
        ut = int(time.time())
        params_name = "params_{}_{}_{}_{}.pkl".format(self.max_epochs, self.num_train, self.batch_size, ut)
        self.network.save_params(params_name)
        self.params_name = params_name
        print("Saved Network Parameters as '{}'!".format(params_name))

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
        plt.show()

    # Output is expected as list, ['U+304A','U+304A','U+304A']
    def predict(self):
        sheet = self.testdata.getSheet()
        test_len = len(self.testdata)

        # データの読み込み
        (x_test, t_test) = set_data(self.testdata, "testdata", test_len, self.filter, self.img_size)

        # 処理に時間のかかる場合はデータを削減 for debug
        # x_test, t_test = x_test[:100], t_test[:100]

        # パラメータの読み込み
        self.network.load_params(self.params_name)

        y = self.network.predict_3_char(x_test)
        y = np.argmax(y, axis=1)
        # ここでy成形する処理を書く
        for i in range(test_len):
            sheet.iloc[i, 1:4] = (['U+0000'], ['U+0000'], ['U+0000'])

        # save predicted results in CSV
        # Zip and submit it.
        sheet.to_csv('test_prediction.csv', index=False)
