# Internship SL: Object Detection in Technical Images using Deep Learning
Code used for internship at [Sioux Lime](https://www.mathware.nl/) on image segmentation using U-Net architecture [1] for two applications: identification of asbestos in electron microscopy images and identification of train defects. 

Data is confidential and not available. 

# Preprocessing
Contrast Limited Histogram Equalization pre-processing and cropping of large images into parts. 
# Training
Techniques explored 
- Segmentation using U-Net
- Data augmentation (horizontal and vertical flipping depending on the application)
- Hard example mining

The methods' performance was evaluated using Precision, Recall and F1 score. 


![image](https://user-images.githubusercontent.com/57133495/141357162-691fa62e-488a-45d6-81e2-f3e6cd44887d.png)


# References

[1] O. Ronneberger, P.Fischer, and T. Brox. U-Net: Convolutional Networks for Biomedical Image Segmentation. In Medical Image Computing and Computer-Assisted Intervention (MICCAI), volume 9351 of LNCS, pages 234–241. Springer, 2015.
