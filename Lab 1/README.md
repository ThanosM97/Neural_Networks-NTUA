# Supervised learning

![UCI ML Logo](http://www.analyticsbodhi.com/wp-content/uploads/2016/03/UCI.png)

For this lab exercise we were given two datasets taken by UCI Machine Learning Repository. The goal of the exercise was to apply different pre-processing techniques and tune the hyperparameters of the machine learning algorithms, using the 10-fold and 5-fold cross validation method with grid search, in order to evaluate the performance of these algorithms on the dataset. 

## LSVT Voice Rehabilitation Data Set
The first dataset we examined is the [LSVT Voice Rehabilitation Data Set](https://archive.ics.uci.edu/ml/datasets/LSVT+Voice+Rehabilitation). It contains data produced via signal processing of the participants voice in order to assess whether voice rehabilitation treatment lead to phonations considered 'acceptable' or 'unacceptable'.

### Dataset information:
- Number of instances: 126
- Number of attributes: 310
- Missing values: No
- Associated tasks: Binary classification
- Labels: 1 for "acceptable", 2 for "unacceptable"

### Pre-processing methods used:
- Variance threshold for feature selection
- Principal components analysis for feature extraction
- Z-score for normalization
- Oversampling to balance the data

### Machine learning algorithms used:
- Dummy classifiers
  - Uniform strategy
  - Constant 1 strategy
  - Constant 2 strategy
  - Most frequent strategy
  - Stratified strategy
 - k Nearest Neighbors algorithm
 
 ### Hyperparameters:
 - Variance threshold
 - Number of components for PCA
 - Number of neighbors for kNN
 
 ## Polish companies bankruptcy dataset
The second dataset of this project is [Polish companies bankruptcy data](http://archive.ics.uci.edu/ml/datasets/Polish+companies+bankruptcy+data). It contains data about bankruptcy prediction of Polish companies. The data was collected from Emerging Markets Information Service (EMIS), which is a database containing information on emerging markets around the world. The bankrupt companies were analyzed in the period 2000-2012, while the still operating companies were evaluated from 2007 to 2013. It includes 5 cases where each demonstrate samples of bankrupty  and contains some instances that represents bankrupted companies and other firms that did not bankrupt in the forecasting period. 

### Dataset information:
- Number of instances: 10503
- Number of attributes: 64
- Missing values: Yes
- Associated tasks: Classification
- Labels: 0 for " Not bankrupted", 1 for "Bankrupted"

**Note**: The original dataset consists of five smaller datasets. Each of those datasets contains data about bankruptcy prediction after a specific period of time. For example, the first dataset contains data about companies that went (or didn't go) bankrupt one year after the data collection. The second dataset about companies that went bankrupt two years after the data collection, and so on. For the purposes of this exercise, we concatenated these five datasets into a single dataset and we didn't use the information about the time period passed from data collection to bankruptcy. 

### Pre-processing methods used:
- Variance threshold for feature selection
- Principal components analysis for feature extraction
- Z-score for normalization
- Oversampling\Undersampling to balance the data

### Machine learning algorithms used:
- Dummy classifiers
  - Uniform strategy
  - Constant 1 strategy
  - Constant 2 strategy
  - Most frequent strategy
  - Stratified strategy
 - k Nearest Neighbors algorithm
 - Gaussian Naive Bayes classifier
 - Multi-Layer Perceptron
 
 ### Hyperparameters:
 - Variance threshold
 - Number of components for PCA
 - Number of neighbors, metric and weights for kNN
 - Hidden layer size, activation, solver, maximum iterations, learning rate and alpha rate for MLP
