

# Image Classification Competition

This repository contains the implementation and exploration of various machine learning models for a 5-class image classification task.

## Project Structure

The code is organized into the following main files:

### Core Implementation Files

- **`logistic_regression_baseline.py`**: Contains the implementation of logistic regression from scratch using One-vs-Rest (OvR) strategy with batch gradient descent. This serves as the baseline model for the competition.

- **`preprocessing.py`**: Contains all preprocessing functions applied to the images, including:
  - Grayscale conversion
  - Flattening (28×28 → 784 dimensions)
  - Normalization (scaling to [0,1])
  - Feature-wise standardization (z-score normalization)

### Main Exploration Notebook

- **`exploration.ipynb`**: The main notebook where all components are integrated and experiments are conducted. This notebook includes:
  - Data loading and visualization
  - Preprocessing pipeline application
  - Training the baseline logistic regression model
  - Hyperparameter tuning 
  - Data augmentation experiments 
  - PCA dimensionality reduction attempts
  - Class weighting strategies
  - **scikit-learn model exploration**:
    - Random Forest Classifier
    - Gradient Boosting Classifier
    - Support Vector Machine (SVM)
    - K-Nearest Neighbors (KNN)
  - Model comparison and evaluation
  - Generation of submission files for Kaggle

### Deep Learning Implementation

- **`cnn_from_scratch.ipynb`**: A separate notebook containing the implementation of a Convolutional Neural Network (CNN) from scratch. This notebook is designed to run in **Google Colab** and provides a deep learning approach to the image classification task.

## How to Use

1. **Install dependencies**: Ensure you have the required libraries (numpy, scikit-learn, matplotlib, seaborn, pandas)

2. **Run preprocessing**: The preprocessing functions in `preprocessing.py` are automatically imported in `exploration.ipynb`

3. **Explore models**: Open `exploration.ipynb` to see all experiments, model comparisons, and results

4. **CNN implementation**: Upload `cnn_from_scratch.ipynb` to Google Colab for the deep learning implementation
>>>>>>> 26ba25f (Image classification project)
