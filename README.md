This project is done in an anaconda2 (4.3.1) environment

Additional packages which are not in anaconda2 are:
Xgboost(0.6)
LightGBM(0.1)
keras(2.0.3)

The competition url is:                    https://www.kaggle.com/c/two-sigma-connect-rental-listing-inquiries

The data is at:                            https://www.kaggle.com/c/two-sigma-connect-rental-listing-inquiries/data

and the "listing_image_time.csv" is from:  https://www.kaggle.com/c/two-sigma-connect-rental-listing-inquiries/discussion/31870

#====================================================================================================================================
project-report:

project report.pdf

#-------------------------------------------------------------------------------------------------------------------------------------

jupyter notebooks for feature building and models:

feature-construction.ipynb                Constructing features that would be used in the base models.

feature-EDA.ipynb                         Exploratory data analysis, the plots are used in the project report.

feature-set-validation-on-lgbm.ipynb      Several runnings validating the effect of the constructed feature sets as decribed in 4 in the report.

BaseModel-ANN.ipynb                       Running a artificial neural network using keras(2.0.3), as described in 5.6

BaseModel-extraTrees&randomForests.ipynb  Running a random forest model and a extra trees model, as described in 5.3 and 5.4

BaseModel-GradientBoosting.ipynb	  Running gradient boosting models by Xgboost(0.6) and LightGBM(0.1), as described in 5.2

BaseModel-knn.ipynb                       Running knn models, as described in 5.1

BaseModel-LogisticRegression.ipynb        Running logistic regression models, as decribed in 5.5

generating-y-for-meta-learner.ipynb       Generating corresponding label series for the meta learner

Stacking.ipynb                            Stacking the result of the base models and generating the output

#--------------------------------------------------------------------------------------------------------------------------------------

python script for tool functions:

mochi.py                                  The scipt is including the functions for constructing some features and running the XGB and LGBM

#--------------------------------------------------------------------------------------------------------------------------------------

others:

stack-beta-0.02eta-3mdsb7cb7-project.csv  Final output file

2sigma-archive.zip                        The archive storing un-organized scripts used in the competition# 2sigma_sorted_version
