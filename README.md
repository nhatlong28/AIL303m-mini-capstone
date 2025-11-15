# ML-Based Tri-Modal Analysis Approach for Medical Insurance

## Project Overview

This project implements a comprehensive machine learning pipeline for analyzing and predicting insurance-related metrics using the insurance dataset. The project encompasses exploratory data analysis, supervised learning for both regression and classification tasks, and unsupervised learning for customer segmentation.

## Project Structure

```
mini_capstone/
├── data/
│   ├── insurance.csv                          # Original dataset
│   ├── insurance_cleaned.csv                  # Preprocessed dataset
│   ├── example_cleaned_classification.csv     # Classification task data
│   └── example_cleaned_regression.csv         # Regression task data
├── notebooks/
│   ├── EDA_and_preprocessing.ipynb           # Exploratory data analysis
│   ├── supervised.ipynb                       # Supervised learning models
│   └── unsupervised.ipynb                     # Unsupervised learning models
├── reports/                                   # Generated reports and outputs
├── requirements.txt                           # Python dependencies
└── README.md                                  # Project documentation
```

## Objectives

### 1. Exploratory Data Analysis and Preprocessing
- Perform comprehensive data quality assessment including missing value and duplicate detection
- Conduct univariate and multivariate analysis of numerical and categorical features
- Implement data preprocessing pipeline with feature scaling and categorical encoding
- Analyze feature distributions and relationships using statistical visualizations

### 2. Supervised Learning
- **Regression Task**: Predict insurance charges using multiple regression algorithms including Linear Regression, Ridge, Lasso, ElasticNet
- **Classification Task**: Predict smoking status using classification algorithms such as Logistic Regression, K-Nearest Neighbors, Support Vector Machines, Decision Trees, Random Forest, Gradient Boosting, and Stacking Classifier
- Handle class imbalance using SMOTE oversampling technique
- Perform hyperparameter tuning with GridSearchCV for model optimization
- Evaluate models using cross-validation with multiple performance metrics

### 3. Unsupervised Learning
- Implement clustering algorithms including K-Means, DBSCAN, and Agglomerative Clustering
- Determine optimal number of clusters using elbow method and silhouette analysis
- Evaluate clustering quality using Silhouette Score, Davies-Bouldin Index, and Calinski-Harabasz Score
- Apply dimensionality reduction with PCA for visualization
- Analyze customer segments based on age, BMI, and number of children

## Dataset Description

The insurance dataset contains the following features:
- **age**: Age of the policyholder
- **sex**: Gender of the policyholder (male/female)
- **bmi**: Body Mass Index
- **children**: Number of dependents covered by insurance
- **smoker**: Smoking status (yes/no)
- **region**: Geographic region (northeast/northwest/southeast/southwest)
- **charges**: Medical insurance costs billed by insurance company

## Technologies and Libraries

- **Data Manipulation**: NumPy, Pandas
- **Visualization**: Matplotlib, Seaborn
- **Machine Learning**: Scikit-learn
- **Imbalanced Learning**: imbalanced-learn
- **Statistical Analysis**: SciPy

## Installation

Install required dependencies using pip:

```bash
pip install -r requirements.txt
```

## Usage

Execute notebooks in the following sequence:

1. **EDA_and_preprocessing.ipynb**: Understand data characteristics and prepare data for modeling
2. **supervised.ipynb**: Train and evaluate regression and classification models
3. **unsupervised.ipynb**: Perform clustering analysis and customer segmentation

## Key Features

- Robust data preprocessing pipeline with feature engineering
- Cross-validation framework for reliable model evaluation
- Comprehensive performance metrics for regression and classification tasks
- Handling of imbalanced datasets using SMOTE
- Model comparison and selection based on multiple evaluation criteria
- Clustering evaluation using multiple validity indices
- Interactive visualizations for model interpretation and cluster analysis

## Presentation

Video presentation available at: [Google Drive](https://drive.google.com/file/d/107jpYStwH4eEYmHTIo58iYAaWHM0t83e/view?usp=sharing)

## License

This project is developed for educational and research purposes.
