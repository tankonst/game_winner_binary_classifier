# Predicting the winner
 Kaggle in-class competition
 
The game has two teams: Radiant and Dire. In these notebooks, we predict the winner. This is a classification task which returns 1 for the prediction that Radiant wins and 0 for the prediction that Dire wins. We aim to maximize the metric ROC AUC for the Kaggle competition.

The files here are:

1. `Explorative Data Analysis.ipynb`

2. `Data_Transformation.ipynb`

3. `Logistic_Regression_model.ipynb`:
 * Load the transformed data
 * Separate categorical features (for one-hot encoding), numerical features (for scaling)
 * Pipeline: feature transformations and logistic regression
 * Fit logistic regression model and save predictions
 * To be added: grid search CV


4. `Stacked_models.ipynb`
To be added
