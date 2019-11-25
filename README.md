# Predicting the winner
Kaggle in-class competition

The game has two teams: Radiant and Dire. In these notebooks, we predict the winner. This is a classification task which returns 1 for the prediction that Radiant wins and 0 for the prediction that Dire wins. We aim to maximize the metric ROC AUC for the Kaggle competition.

The original data provided by the organizers has 39675 sample and 245 features. Additional features were extracted from provided metadata (.json files).

The repository contains next files:

1. `Explorative Data Analysis.ipynb`: EDA
  * Plot distributions of all features, notice some features are skewed.
  * Plot coordinates by team. When Radiant wins, density of Dire players in Radiant corner is low. This can be used to create a density feature.
  * Frequency plots with data separated by class

2. `Data_Transformation.ipynb`: feature engineering
  * Build features on hero IDs. Target-encoding hero IDs in two ways:
    +  First, for each hero, take # games won - # games lost. This new feature also measures hero popularity and may not be reflective of "normalized" success rates.
    + For the second feature, take (# games won - # games lost)/total games played.
   * Transform health: add features representing number of deaths per team and also average health per team
   * Transform coordinates: add indicator function for each player with 1 if in x team's base, 0 otherwise.
   * Aggregate features for each team: take sum over all team players, standard deviation over all team players, over levels.
   * Feature transformations: log and square of selected features

3. `Logistic_Regression_model.ipynb`:
 * Load the transformed data
 * Separate categorical features (for one-hot encoding), numerical features (for scaling)
 * Pipeline: feature transformations and logistic regression
 * Grid search CV
 * Fit logistic regression model for best parameters and save predictions


4. `Stacked_models.ipynb`
    To be added

5. `Feature_Selection_RFECV.ipynp`
    Selective Feature Extraction with Cross Validation for identifying the optimal number of features for Logistic Regression. The resulting model on a smaller subset provides the same score the full set of features.
