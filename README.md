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
No_Attention|Fold 1|Meningioma|0.9642|0.8503|0.9036
No_Attention|Fold 1|Glioma|0.9755|0.9715|0.9734
No_Attention|Fold 1|Pituitary|0.9135|1.0|0.9547
No_Attention|Fold 2|Meningioma|0.9401|0.9515|0.9457
No_Attention|Fold 2|Glioma|0.9852|0.9460|0.9457
No_Attention|Fold 2|Pituitary|0.9195|0.9876|0.9523
No_Attention|Fold 3|Meningioma|0.9568|0.8986|0.9267
No_Attention|Fold 3|Glioma|0.9437|0.9775|0.9593
No_Attention|Fold 3|Pituitary|0.9900|0.9950|0.9924
No_Attention|Fold 4|Meningioma|0.9112|0.9482|0.9293
No_Attention|Fold 4|Glioma|1.0|0.9759|0.9878
No_Attention|Fold 4|Pituitary|0.9664|0.9719|0.9691
No_Attention|Fold 5|Meningioma|0.8674|0.9350|0.8999
No_Attention|Fold 5|Glioma|0.9651|0.9928|0.9787
No_Attention|Fold 5|Pituitary|0.9947|0.9000|0.9449
CAB|Fold 1|Meningioma|0.9910|0.9098|0.9486
CAB|Fold 1|Glioma|0.9836|0.9917|0.9876
CAB|Fold 1|Pituitary|0.9567|1.0|0.9778
CAB|Fold 2|Meningioma|0.9281|0.9281|0.9281
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
PAB|Fold 5|Pituitary|0.9947|0.9473|0.9473
No_Split|Fold 1|Meningioma|0.9821|0.9401|0.9606
No_Split|Fold 1|Glioma|0.9759|0.9836|0.9797
No_Split|Fold 1|Pituitary|0.9783|1.0|0.9890
No_Split|Fold 2|Meningioma|0.9461|0.9575|0.9517
No_Split|Fold 2|Glioma|0.9822|0.9485|0.9650
No_Split|Fold 2|Pituitary|0.9252|0.9817|0.9526
No_Split|Fold 3|Meningioma|0.9712|0.8823|0.9246
No_Split|Fold 3|Glioma|0.9264|0.9907|0.9574
No_Split|Fold 3|Pituitary|0.9950|0.9901|0.9925
No_Split|Fold 4|Meningioma|0.8870|0.9482|0.9165
No_Split|Fold 4|Glioma|0.9969|0.9700|0.9832
No_Split|Fold 4|Pituitary|0.9608|0.9662|0.9634
No_Split|Fold 5|Meningioma|0.8734|0.9235|0.8978
No_Split|Fold 5|Glioma|0.9547|0.9891|0.9715
No_Split|Fold 5|Pituitary|0.9947|0.9043|0.9473
Ours|Fold 1|Meningioma|1.0|0.9105|0.9531
Ours|Fold 1|Glioma|0.9755|0.9958|0.9855
Ours|Fold 1|Pituitary|0.9675|1.0|0.9834
Ours|Fold 2|Meningioma|0.9640|0.9515|0.9577
Ours|Fold 2|Glioma|0.9852|0.9526|0.9686
Ours|Fold 2|Pituitary|0.9675|0.9938|0.9804
Ours|Fold 3|Meningioma|0.9712|0.8881|0.9277
Ours|Fold 3|Glioma|0.9350|0.9863|0.9599
Ours|Fold 3|Pituitary|0.9900|0.9950|0.9924
Ours|Fold 4|Meningioma|0.9112|0.9576|0.9338
Ours|Fold 4|Glioma|0.9969|0.9759|0.9862
Ours|Fold 4|Pituitary|0.9720|0.9775|0.9747
Ours|Fold 5|Meningioma|0.8674|0.9473|0.9055
Ours|Fold 5|Glioma|0.9686|0.9858|0.9771
Ours|Fold 5|Pituitary|0.9947|0.9043|0.9473
