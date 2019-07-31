import numpy as np
import matplotlib.pyplot as plt
from srcs.convnet import ConvNet
from srcs.functions import set_data
from srcs.trainer import Trainer
import time
from PIL import Image


class MyAlgorithm():
    """
    Build your algorithm.
    """
    def build_model(self, traindata, valdata, max_epochs, num_train, batch_size, img_size):
        filter = Image.open("filter.jpg").convert('L')
        filter = np.array(filter.resize((img_size, img_size)))

        # データの読み込み
        (x_train, t_train) = set_data(traindata, "traindata", num_train, filter, img_size)
        (x_val, t_val) = set_data(valdata, "valdata", num_train, filter, img_size)

        # 処理に時間のかかる場合はデータを削減
        # x_train, t_train = x_train[:5000], t_train[:5000]
        # x_val, t_val = x_val[:1000], t_val[:1000]

        print("set model ...")
        network = ConvNet(input_dim=(1, img_size, img_size),
                          conv_param={'filter_num': 30, 'filter_size': 5, 'pad': 0, 'stride': 1},
                          hidden_size=100, output_size=48, weight_init_std=0.01)

        trainer = Trainer(network, x_train, t_train, x_val, t_val,
                          epochs=max_epochs, mini_batch_size=batch_size,
                          optimizer='Adam', optimizer_param={'lr': 0.001},
                          evaluate_sample_num_per_epoch=900)
        print("train ...")
        trainer.train()

        # パラメータの保存
        ut = int(time.time())
        params_name = "params_{}_{}_{}_{}.pkl".format(max_epochs, num_train, batch_size, ut)
        network.save_params(params_name)
        print("Saved Network Parameters as '{}'!".format(params_name))

        # グラフの描画
        markers = {'train': 'o', 'test': 's'}
        x = np.arange(max_epochs)
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
    def predict(self, img):
        img = np.array(img.convert('L'))
        feat = np.resize(img, 768)  # feature vector
        feat = np.reshape(feat, (3, 256))
        dist = [0] * 3
        y_pred = ['U+0000'] * 3
        for i in range(3):
            dist[i] = np.linalg.norm(self.model - feat[i], axis=1)  # measure distance
            y_pred[i] = self.y_train[np.argmin(dist[i])]  # Get the closest
        return y_pred
