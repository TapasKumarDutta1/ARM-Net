## General Overview
Implemented HDenseUNet for segmentation. Trained using federated learning approach on LITS dataset and tested on 3Ddirac dataset. Augmented using WGAN.


## WGAN Examples


IMage          |  Mask
:-------------------------:|:-------------------------:
![](image_2.png)  |  ![](seg_2.png)
![](image_1.png)  |  ![](seg_1.png)
![](image.png)  |  ![](seg.png)

## Results

Dice          |  target
:-------------------------:|:-------------------------:
94.6  |  Liver
54  |  Tumor


## References
- H-DenseUNet: Hybrid Densely Connected UNet for Liver and Tumor Segmentation from CT Volumes, TMI 2018 Xiaomeng Li, Hao Chen, Xiaojuan Qi, Qi Dou, Chi-Wing Fu, Pheng-Ann Heng
- TOWARDS FEDERATED LEARNING AT SCALE: SYSTEM DESIGN Keith Bonawitz 1 Hubert Eichner 1 Wolfgang Grieskamp 1 Dzmitry Huba 1 Alex Ingerman 1 Vladimir Ivanov 1
Chloe Kiddon ´
1 Jakub Konecnˇ y´
1 Stefano Mazzocchi 1 H. Brendan McMahan 1 Timon Van Overveldt 1
David Petrou 1 Daniel Ramage 1 Jason Roselander 
- Wasserstein GAN Martin Arjovsky, Soumith Chintala, Léon Bottou
