import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image


# Utility functions
def unicode_to_kana(code: str):
    # Example: unicode_to_kana('U+304A')
    assert len(code) == 6
    return chr(int(code[2:], 16))


def unicode_to_kana_list(codes: list):
    # Example: unicode_to_kana_list( ['U+304A','U+304B','U+304D'] )
    assert len(codes) == 3
    return [unicode_to_kana(x) for x in codes]


def kana_to_unicode(kana: str):
    assert len(kana) == 1
    return 'U+' + hex(ord(kana))[2:]


def evaluation(y0, y1):
    cols = ['Unicode1', 'Unicode2', 'Unicode3']
    x = y0[cols] == y1[cols]
    x2 = x['Unicode1'] & x['Unicode2'] & x['Unicode3']
    acc = sum(x2) / len(x2) * 100
    # n_correct = (np.array(y0[cols]) == np.array(y1[cols])).sum()
    # acc = n_correct / (len(y0) * 3) * 100
    return acc


def code2index(code):
    codes = ["U+3042", "U+3044", "U+3046", "U+3048", "U+304A", "U+304B", "U+304D", "U+304F",
             "U+3051", "U+3053", "U+3055", "U+3057", "U+3059", "U+305B", "U+305D", "U+305F",
             "U+3061", "U+3064", "U+3066", "U+3068", "U+306A", "U+306B", "U+306C", "U+306D", "U+306E", "U+306F",
             "U+3072", "U+3075", "U+3078", "U+307B", "U+307E", "U+307F",
             "U+3080", "U+3081", "U+3082", "U+3084", "U+3086", "U+3088", "U+3089", "U+308A", "U+308B", "U+308C", "U+308D", "U+308F",
             "U+3090", "U+3091", "U+3092", "U+3093"]
    return codes.index(code)


# This class manages all targets: train, val, and test
class AlconTargets():
    """
    This class load CSV files for train and test.
    It generates validation automatically.

    Arguments:
       + datapath is string designate path to the dataset, e.g., './dataset'
       + train_ratio is a parameter for amount of traindata.
         The remain will be the amount of validation.
    """
    def __init__(self, datapath: str, train_ratio: float):
        self.datapath = Path(datapath)

        # Train annotation
        fnm = Path(datapath) / Path('train') / 'annotations.csv'
        assert fnm.exists()
        df = pd.read_csv(fnm).sample(frac=1)

        # Split train and val
        nTrain = round(len(df) * train_ratio)
        self.train = df.iloc[0:nTrain]
        self.val = df.iloc[nTrain:]

        # Test annotation
        fnm = Path(datapath) / Path('test') / 'annotations.csv'
        assert fnm.exists()
        self.test = pd.read_csv(fnm)


class AlconDataset():
    """
    This Dataset class provides an image and its unicodes.

    Arguments:
       + datapath is string designate path to the dataset, e.g., './dataset'
       + targets is DataFrame provided by AlconTargets, e.g., AlconTargets.train
       + isTrainVal is boolean variable.
    """
    def __init__(self, datapath: str, targets: 'DataFrame', isTrainVal: bool):
        # Targets
        self.targets = targets

        # Image path
        if isTrainVal:
            p = Path(datapath) / Path('train')
        else:
            p = Path(datapath) / Path('test')
        self.img_path = p / 'imgs'  # Path to images

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx: int):
        # Get image リサイズ、2値化
        ident = self.targets['ID'].iloc[idx]  # ID
        img_fnm = self.img_path / (str(ident) + '.jpg')  # image filename
        assert img_fnm.exists()
        img_size = 32
        img = Image.open(img_fnm)
        img = img.convert('L')
        img = np.array(img.resize((img_size, img_size * 3)))
        img = (img > 120) * 255

        # Get annotations
        unicodes = list(self.targets.iloc[idx, 1:4])
        unicodes = np.array([code2index(unicodes[0]), code2index(unicodes[1]), code2index(unicodes[2])])

        return  img, unicodes

    def showitem(self, idx: int):
        img, codes = self.__getitem__(idx)
        print(unicode_to_kana_list(codes))
        img.show()

    # You can fill out this sheet for submission
    def getSheet(self):
        sheet = self.targets.copy()  # Deep copy
        sheet[['Unicode1', 'Unicode2', 'Unicode3']] = None  # Initialization
        return sheet
