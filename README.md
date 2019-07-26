<<<<<<< HEAD
# PRMU challenge on old Japanese character recognition, 2019
This project provides source codes and dataset for recognizing successive three characters in old Japanese documents, and output unicodes of a set of three characters.

+ [Official Website](https://sites.google.com/view/alcon2019) in Japanese.
+ Samples: A rectangle including successive three characters. Three characters appear vertically in the rectangle.
+ Categories: 48 of KANA (Japanese alphabets), called "Kuzushiji". Not including KANJI (Chinese characters).
+ References: Refer to [CODH seminars](http://codh.rois.ac.jp/seminar/) for classical Japanese literatures, [KMNIST](https://github.com/rois-codh/kmnist) for Kuzushiji characters in classical Japanese literatures, [Kuzushiji Dataset](http://codh.rois.ac.jp/char-shape/) for the details of the dataset.


# Quick Start
1. Install required packages: PIL, pandas, skimage, numpy.
2. Run example script by `python main.py`
3. The results are saved in test_prediction.csv


# Dataset
The datasset of this competition is build on [Kuzushiji Dataset](http://codh.rois.ac.jp/char-shape/) provided by [ROIS-DS Center for Open Data in the Humanities (CODH)](http://codh.rois.ac.jp/index.html.en). The dataset inherits [Creative Commons by ShareAlike 4.0 International (CC BY-SA)](https://creativecommons.org/licenses/by-sa/4.0/)

**Please indicate DOI of Kuzushiji Dataset if you refer to the dataset.**  

+ Plain text:  
> Kuzushiji Dataset, National Institute of Japanese Literature and ROIS-DS Center for Open Data in the Humanities, doi:10.20676/00000340, 2016. URL: http://codh.rois.ac.jp/char-shape/ Accessed: [Insert date here]

+ Bibtex  
> @misc{alcon2019prmu,  
author = {National Institute of Japanese Literature and ROIS-DS Center for Open Data in the Humanities},  
title = {Kuzushiji Dataset},  
year = {2016},  
howpublished = {\url{http://codh.rois.ac.jp/char-shape/}},  
note = {Accessed: [Insert date here]}  
}


# Organizers
+ Tomo Miyazaki (Tohoku University)
+ Toru Tamaki (Hiroshima University)
+ Kazuaki Nakamura (Osaka Univerity)
+ Masashi Nishiyama (Tottori University)
+ Yusuke Uchida (DeNA)
+ Takanori Ogata (ABEJA)
+ Keiichiro Shirai (Sinshu University)
+ Asanobu Kitamoto (ROIS-DS Center for Open Data in the Humanities, NII)
+ Tarin Clanuwat (ROIS-DS Center for Open Data in the Humanities, NII)

# Supported by
+ IEICE-ISS Technical Committee on [Pattern Recognition and Media Understanding (PRMU)](http://www.ieice.org/~prmu/jpn/)
+ [ROIS-DS Center for Open Data in the Humanities (CODH)](http://codh.rois.ac.jp/index.html.en)

# Inquiries
For any inquiries you may have regarding the competition, please contact the competition office via:
E-Mail: prmu-fy2019alcon@mail.ieice.org
=======
# RecognizeOldCharacter
23th PRMU Algorithm Contest - Recognize 3 consecutive old Japanese characters
>>>>>>> origin/master
