
# Image Classification Competition

This repository documents a structured exploration and evaluation of classical machine learning and deep learning models for a five-class image classification task.

This project was developed within an academic machine learning competition context, focused on comparative evaluation of classification models.

---

## Project Structure

The codebase is organized into the following main components:

### Core Implementation Files

- **`logistic_regression_baseline.py`**  
  Implementation of logistic regression from scratch using a One-vs-Rest (OvR) strategy and batch gradient descent.  
  This model serves as the primary baseline for the project.

- **`preprocessing.py`**  
  Preprocessing utilities applied to the image data, including:
  - Grayscale conversion
  - Flattening (28×28 → 784 dimensions)
  - Pixel normalization to the [0, 1] range
  - Feature-wise standardization (z-score normalization)

---

### Main Exploration Notebook

- **`exploration.ipynb`**  
  The main experimental notebook integrating data processing, model training, and evaluation.  
  It includes:
  - Data loading and visualization
  - Application of the preprocessing pipeline
  - Training and tuning of the baseline logistic regression model
  - Data augmentation experiments
  - PCA-based dimensionality reduction attempts
  - Class imbalance handling strategies
  - Exploration of scikit-learn models:
    - Random Forest Classifier
    - Gradient Boosting Classifier
    - Support Vector Machine (SVM)
    - K-Nearest Neighbors (KNN)
  - Model comparison and evaluation
  - Generation of Kaggle submission files

---

### Deep Learning Implementation

- **`cnn_from_scratch.ipynb`**  
  A standalone notebook implementing a Convolutional Neural Network (CNN) from scratch.  
  Designed to run in **Google Colab**, this notebook explores a deep learning approach while preserving the spatial structure of the images.

---

## Results

- Multiple classical and deep learning models were evaluated on a constrained, imbalanced five-class image dataset
- Classical machine learning models provided strong and stable baselines under limited data conditions
- Increased model complexity did not consistently translate into improved generalization
- The project achieved competitive performance on the Kaggle leaderboard

---

## Key Learnings

- Trade-offs between classical machine learning and deep learning for small and imbalanced image datasets
- Importance of establishing strong baselines before introducing complex architectures
- Practical challenges of training CNNs from scratch with limited data
- Impact of preprocessing choices and class imbalance on model performance and generalization

---

## How to Use

1. **Install dependencies**  
   Ensure the required libraries are installed:
```

numpy
pandas
matplotlib
seaborn
scikit-learn

```

2. **Run preprocessing**  
The preprocessing functions defined in `preprocessing.py` are automatically imported in `exploration.ipynb`.

3. **Explore models and experiments**  
Open `exploration.ipynb` to reproduce experiments, compare models, and analyze results.

4. **Run the CNN implementation**  
Upload `cnn_from_scratch.ipynb` to Google Colab to execute the deep learning experiments.

---

## Academic Context

This project was developed as part of **IFT6390 – Fondements de l’apprentissage machine**  
at the **Université de Montréal**, and corresponds to the second competition of the course.

The project achieved a **13th place ranking** on the Kaggle leaderboard.

---

## Technical Report

A detailed technical report describing the experimental setup, model comparisons, and analysis
is available in the `report/` directory.

>>>>>>> 
