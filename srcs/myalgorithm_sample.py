import numpy as np


class MyAlgorithm():
    """
    Build your algorithm.
    """
    def build_model(self, traindata):
        # Initialization
        num_train = 100
        model = np.zeros((num_train, 1024))
        y_train = [['U+0000'] * 3 for i in range(num_train)]

        # Convert images to features
        for i in range(num_train):
            img, codes = traindata[i]  # Get an image and label
            img = np.array(img.convert('L'))  # Gray scale
            model[i, :] = np.resize(img, 1024)  # feature vector
            y_train[i] = codes

        # Keep model and labels
        self.model = model
        self.y_train = y_train

    # Output is expected as list, ['U+304A','U+304A','U+304A']
    def predict(self, img):
        img = np.array(img.convert('L'))
        feat = np.resize(img, 1024)  # feature vector
        dist = np.linalg.norm(self.model - feat, axis=1)  # measure distance
        y_pred = self.y_train[np.argmin(dist)]  # Get the closest
        print("dist: " + str(np.argmin(dist)))
        return y_pred
