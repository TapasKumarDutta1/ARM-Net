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

Attention Used|Fold|Class| Precision | Recall | F1-score
---|---|---|---|---|---
CAB|Fold 1|Meningioma|0.9910|0.9098|0.9486
CAB|Fold 1|Glioma|0.9836|0.9917|0.9876
CAB|Fold 1|Pituitary|0.9567|1.0|0.9778

CAB|Fold 2|Meningioma|0.9281|0.9281|1.0
CAB|Fold 2|Glioma|0.9704|0.9507|0.9604
CAB|Fold 2|Pituitary|0.9540|0.9940|0.9735

CAB|Fold 3|Meningioma|0.9568|0.8986|0.9267
CAB|Fold 3|Glioma|0.9350|0.9773|0.9556
CAB|Fold 3|Pituitary|1.0|0.9950|0.9974

CAB|Fold 4|Meningioma|0.8951|0.9652|0.9288
CAB|Fold 4|Glioma|0.9907|0.9611|0.9756
CAB|Fold 4|Pituitary|0.9776|0.9831|0.9803

CAB|Fold 5|Meningioma|0.8734|0.9539|0.8645
CAB|Fold 5|Glioma|0.9721|0.9893|0.9806
CAB|Fold 5|Pituitary|0.9947|0.9043|0.9473

PAB|Fold 1|Meningioma|0.9910|0.9250|0.9568
PAB|Fold 1|Glioma|0.9795|0.9836|0.9815
PAB|Fold 1|Pituitary|0.9621|1.0|0.9806

PAB|Fold 2|Meningioma|0.9454|0.9341|0.9392
PAB|Fold 2|Glioma|0.9852|0.9380|0.9610
PAB|Fold 2|Pituitary|0.9022|0.9874|0.9428

PAB|Fold 3|Meningioma|0.9640|0.8758|0.9177
PAB|Fold 3|Glioma|0.9264|0.9771|0.9510
PAB|Fold 3|Pituitary|0.9900|1.0|0.9949

PAB|Fold 4|Meningioma|0.9112|0.9262|0.9186
PAB|Fold 4|Glioma|0.9969|0.9759|0.9862
PAB|Fold 4|Pituitary|0.9497|0.9770|0.9631

PAB|Fold 5|Meningioma|0.8773|0.9225|0.8993
PAB|Fold 5|Glioma|0.9616|0.9892|0.9752
PAB|Fold 5|Pituitary|0.9947|0.9473
