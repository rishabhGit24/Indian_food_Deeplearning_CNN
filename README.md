Indian Food Classification Using Deep Learning (CNN)
This project focuses on building a deep learning model to classify various types of Indian food using a Convolutional Neural Network (CNN). The model is trained on a dataset of Indian food images and leverages advanced image recognition techniques to accurately identify and classify them.

Table of Contents
Project Overview
Installation
Dataset
Model Architecture

Indian cuisine is diverse, with each region having its unique set of dishes. This project aims to create a CNN-based model that can classify different Indian food items by analyzing images of the food. The model is trained to learn patterns in the images and predict the food category with high accuracy.

Installation:
To get started with the project, follow these steps:

Clone the repository:

git clone https://github.com/rishabhGit24/Indian_food_Deeplearning_CNN.git
cd Indian_food_Deeplearning_CNN
Install the required dependencies:

pip install -r requirements.txt

Dataset:
The dataset used in this project consists of images of various Indian dishes. You can either use an existing Indian food dataset from sources like Kaggle or create your own dataset by collecting images from the web.


Each class corresponds to a type of Indian food, such as "Biryani," "Samosa," "Dosa," etc.

Model Architecture
The model uses a Convolutional Neural Network (CNN) architecture built with TensorFlow or PyTorch. The architecture includes:
1).Convolutional layers for feature extraction
2).MaxPooling layers for downsampling
3).Fully connected layers for classification
4).Softmax activation in the final layer for multiclass classification
