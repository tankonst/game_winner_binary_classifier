# dota_challenge_binary_classifier
 Prediction of winning team (out of two) in a computer game (Kaggle competition).
 The original data provided by the organizers has 39675 sample and 245 features. Additional features were extracted from provided metadata (.json files).

 The repository contains next files:

 * Exploratory_Data_Analysis.ipynb
  EDA of the data set.

* Data_Transformation.ipynb
 Feature engineering for the modeling. The file uses library module transformations.py

* Feature_Selection_RFECV.ipynp
 Selective Feature Extraction with Cross Validation for identifying the optimal number of features for Logistic Regression. The resulting model on a smaller subset provides the same score the full set of features.

 * Logistic_Regression_model.ipynp
 Training of the regularized logistic regression model. Hyperparameter tuning with GridSearchCV.
