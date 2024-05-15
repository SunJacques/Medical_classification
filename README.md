# Dermoscopic Image Classification Challenge

## Objective

The primary goal of this project is to classify dermoscopic images of skin lesions into one of eight diagnostic categories:

1. Melanoma
2. Melanocytic Nevus
3. Basal Cell Carcinoma
4. Actinic Keratosis
5. Benign Keratosis
6. Dermatofibroma
7. Vascular Lesion
8. Squamous Cell Carcinoma

## Methodology

### Image Segmentation

Some images in the dataset are provided with corresponding masks. To utilize this information, I trained a U-Net model from scratch to segment the lesions. 
The training was performed on a dataset comprising 1800 images. The U-Net model achieved a Dice score of 0.87, indicating a high level of accuracy in segmenting the lesions.

### Classification

For the classification task, I applied several classical machine learning models, including Random Forest (RF), Support Vector Machine (SVM), and XGBoost. 
These models were used to classify the segmented lesions into the eight diagnostic classes. The combined approach yielded a weighted accuracy of 0.5.

## Results

- **Segmentation:** U-Net model achieved a Dice score of 0.87.
- **Classification:** Classical ML models achieved a weighted accuracy of 0.5.

## Conclusion

The combination of U-Net for lesion segmentation and classical machine learning models for classification provides a solid foundation for improving the accuracy of skin lesion diagnosis. 
Future work will involve enhancing the classification accuracy by experimenting with pretrained model like ResNet.
