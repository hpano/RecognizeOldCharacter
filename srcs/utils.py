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
        # Get image
        ident = self.targets['ID'].iloc[idx]  # ID
        img_fnm = self.img_path / (str(ident) + '.jpg')  # image filename
        assert img_fnm.exists()
        img = Image.open(img_fnm)
        # Get annotations
        unicodes = list(self.targets.iloc[idx, 1:4])
        return img, unicodes

    def showitem(self, idx: int):
        img, codes = self.__getitem__(idx)
        print(unicode_to_kana_list(codes))
        img.show()

    # You can fill out this sheet for submission
    def getSheet(self):
        sheet = self.targets.copy()  # Deep copy
        sheet[['Unicode1', 'Unicode2', 'Unicode3']] = None  # Initialization
        return sheet
