# Code for thesis simulation and application on epidemiological data

This repository contains the code used in Janet's master thesis. Each file corresponds to one chapter in the thesis. 

"Simulation 1 code" presents the code for a logistic regression model with its coefficients assigned with different priros: the Laplace prior, the Horseshoe prior and the regularized Horseshoe prior. 

"Simulation 2 code" illustrates the code of our proposed hierarchical model. It simultaneously imputes the missing values in one predictor and fits the outcome prediction regression model.  

In the thesis, we use the code in "AS data analysis code" to explore the possible inequality existed in treatment decisions among patients in Quebec who diagnosed with aortic stenosis and needed valve replacement. In addition to the variable selection processed through the regularized Horseshoe prior, we added a binary adjacency structure to model the differences between CLSC regions. 

Finally, in "Missing data imputation with HIV data", we modified the hierarchical model proposed in simulation 2 to impute for two variables and investigated the determinants of HIV pravelence.  
