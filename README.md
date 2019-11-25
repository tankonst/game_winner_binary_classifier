# Predicting the winner
 Kaggle in-class competition

The game has two teams: Radiant and Dire. In these notebooks, we predict the winner. This is a classification task which returns 1 for the prediction that Radiant wins and 0 for the prediction that Dire wins. We aim to maximize the metric ROC AUC for the Kaggle competition.

The original data provided by the organizers has 39675 sample and 245 features. Additional features were extracted from provided metadata (.json files).

 The repository contains next files:

1. `Exploratory_Data_Analysis.ipynb`
    EDA of the data set.

2. `Data_Transformation.ipynb`
    Feature engineering for the modeling. The file uses library module transformations.py .

3. `Feature_Selection_RFECV.ipynp`
    Selective Feature Extraction with Cross Validation for identifying the optimal number of features for Logistic Regression. The resulting model on a smaller subset provides the same score the full set of features.

4. `Logistic_Regression_model.ipynb`
    Training of the regularized logistic regression model. Hyperparameter tuning with GridSearchCV.

5. `Stacked_models.ipynb`
To be added
