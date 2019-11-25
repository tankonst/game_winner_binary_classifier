# Predicting the winner
 Kaggle in-class competition
 
The game has two teams: Radiant and Dire. In these notebooks, we predict the winner. This is a classification task which returns 1 for the prediction that Radiant wins and 0 for the prediction that Dire wins. We aim to maximize the metric ROC AUC for the Kaggle competition.

The files here are:

1. `Explorative Data Analysis.ipynb`

2. `Data_Transformation.ipynb`: feature engineering
  * Build features on hero IDs
    + target-encdoding hero IDs in two ways: first, for each hero, take # games won - # games lost. This new feature also measures hero popularity and may not be reflective of "normalized" success rates. So, for the second feature, take (# games won - # games lost)/total games played.

3. `Logistic_Regression_model.ipynb`:
 * Load the transformed data
 * Separate categorical features (for one-hot encoding), numerical features (for scaling)
 * Pipeline: feature transformations and logistic regression
 * Grid search CV
 * Fit logistic regression model for best parameters and save predictions


4. `Stacked_models.ipynb`
To be added
