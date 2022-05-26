## CDANET
This repository contains the Tensorflow implementation of our paper "CDANET: CHANNEL SPLIT DUAL ATTENTION BASED CNN FOR BRAIN TUMOR
CLASSIFICATION IN MR IMAGES"

## Requirements
Python 3.7.13 <br />
<br />
Numpy 1.21.6 <br />
<br />
Tensorflow 2.8.0 <br />
<br />
Opencv-python 4.1.2 <br />
<br />
Pandas 1.3.5 <br />
<br />
h5py 3.1.0 <br />
<br />
Imgaug 0.2.9 <br />
<br />

## Trained models
- Fold1: https://drive.google.com/file/d/12RYBrUj65dcKA2R7_SZpe1jEjFVyrtCA/view?usp=sharing
- Fold2: https://drive.google.com/file/d/1VXMTRPqOi_6sUQ4e0nfgI3aY4_A_rv2E/view?usp=sharing
- Fold3: https://drive.google.com/file/d/16qaLxYutoOvRDrOfQOdDrV4fgqm4BvJE/view?usp=sharing
- Fold4: https://drive.google.com/file/d/1TM6N_ZywwSvw25emeSYhE_64DpHYQY2i/view?usp=sharing
- Fold5: https://drive.google.com/file/d/1m952RXGEz3dudJTgGW9hAxKagNR9585Y/view?usp=sharing

## Useage
- Clone the repository, and download the weights of the trained model, put them into 'weights-cdanet' folder, you can run test.py to test the model directly. 

- If you want to train the model, download the dataset

- Run the following code: python train.py. Note that parameters and paths should be set beforehand

- To test the model, run test.py once the training is complete. Execute python test.py once training is complete. 

## Brats2020 
Brats is a 3d dataset with 2 different labels 293 HGG and 76 LGG samples. We used a stratified five fold cross validation scheme and trained the entire model in an end to end manner.
 
## Results

Dataset Used|Five Fold Average Accuracy
 --- | ---
Brats 2020 | 0.9653
Figshare | 0.9660

