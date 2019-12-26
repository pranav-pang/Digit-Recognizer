# Digit-Recognizer

## Purpose
My first foray into ML projects, this is a CNN for classifying handwritten digits in the MNIST dataset. This was created for the Kaggle Digit Recognizer competition. As the data files were too large to upload to GitHub, anyone wishing to use this code can download the training and test data directly from Kaggle here: https://www.kaggle.com/c/digit-recognizer/data

## Model Architecture
The model is a Sequential Keras model and consists of two Conv2D layers, followed by a Dropout layer to prevent overfitting, and Dense layers to generate probabilities for each class. The convolutional layers all used ReLU activation functions, while the final output layer utilzes a softmax activation function to generate class probabilities for the image. The input to the model is an array of 28x28 grayscale images and the predictions are output as a nx10 numpy array, where n is the number of images. For a particular row, each column contains the probability that that image is of that particular digit. For example, if the value in row 10, column 8 was 0.85, that would mean that the 10th image has an 85% chance of being of the digit 8.

## Results
Below are the statistics for each epoch during training:

| Epoch | Training Loss | Training Accuracy | Validation Loss | Validation Accuracy |
|-------|---------------|-------------------|-----------------|---------------------|
| 1     | 0.2859        | 0.9154            | 0.1060          | 0.9663              |
| 2     | 0.0812        | 0.9759            | 0.0607          | 0.9783              |
| 3     | 0.0537        | 0.9835            | 0.0535          | 0.9838              |

The model received a test accuracy of 98% upon submission.
