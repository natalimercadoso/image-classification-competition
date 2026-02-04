import pickle 
import numpy as np
import matplotlib.pyplot as plt 

test_path = 'Data/ift-3395-6390-kaggle-2-competition-fall-2025/test_data.pkl'
train_path = 'Data/ift-3395-6390-kaggle-2-competition-fall-2025/train_data.pkl'

def load(path):
    with open(path, 'rb') as file:
        data = pickle.load(file)
    return data

train_data = load(train_path)
test_data = load(test_path)

def get_train_images_labels(data):
    images = data['images']
    labels = data['labels']
    return images, labels

def get_test_images(data):
    return data['images']

# Grayscale
def train_preprocessing(images):
    # Mean in the RGB channel to convert to grayscale 
    images_in_gray = np.mean(images, axis=3)
    N, H, W = images_in_gray.shape
    X = images_in_gray.reshape(N, H * W)
    X = X.astype(np.float32) / 255.0
    mean = np.mean(X, axis=0)
    std  = np.std(X, axis=0) + 1e-8 #to avoid division by 0
    X_norm = (X - mean) / std
    return X_norm, mean, std 

def preprocessing_test(images, mean, std):
    images_in_gray = np.mean(images, axis=3)
    N, H, W = images_in_gray.shape
    X = images_in_gray.reshape(N, H * W)
    X = X.astype(np.float32) / 255.0
    X_norm = (X - mean) / std
    return X_norm



    