# Life-Expectancy-Prediction

## Overview 
The primary goal of this project is to build accurate models for data classification using CNN and KNN. These models are designed to process complex data patterns and make reliable predictions based on the input data.

Dataset Description 
Dataset: https://www.kaggle.com/datasets/kumarajarshi/life-expectancy-who
This dataset is provided by the World Health Organization (WHO). The data is related to various factors that affect life expectancy, these include factors such as mortality factors, economical factors and social factors. These factors are then correlated with life expectancy. The dataset covers a period of 16 years between 2000 to 2015 for 193 countries. The dataset contains a total of 22 columns and 2938 rows.

### Variable Selection

The variables were chosen based on their relevance to predicting life expectancy. To ensure the algorithms performed well, careful preprocessing was applied.

For KNN:

Numerical variables were selected as KNN requires numeric input.

Categorical variables (like economic status and country) were one-hot encoded or label encoded.

Features were normalized using a z-score to ensure consistency during distance calculation.

For CNN:
Normalization was performed to scale the values between 0 and 1 for economic status, and categorical variables (such as country) were encoded to be compatible with the CNN model.

### K nearest neighbour 
K-NN works by using a distance metric to calculate how close a feature is to a k value. The algorithm assumes that the closer they are, the more related in this case the closer the features are to an age of life expectancy the more likely the life expectancy is that age.

### Convolutional Neural Network (CNN)

The CNN model is used for tasks requiring spatial data analysis, particularly image classification but it can also be adapted for regression tasks by normalising the input and changing certain parameters such as the activation function and loss function. The network consists of two convolutional layers followed by a max pooling layer, then flattened and passed through a dense layer. A dropout layer is applied to prevent overfitting, followed by another dense layer to produce the final output.

### Results 
Root Mean Squared Error (RMSE) was used as the primary metric to evaluate the performance of both the CNN and KNN models. RMSE measures the average magnitude of the prediction errors, with lower values indicating better model accuracy and fit to the data.

The CNN and KNN models were evaluated on the dataset, achieving promising accuracy rates. Model performance can be seen from PNG files. 
