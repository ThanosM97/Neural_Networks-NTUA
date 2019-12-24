# Supervised learning

![UCI ML Logo](http://www.analyticsbodhi.com/wp-content/uploads/2016/03/UCI.png)

For this lab exercise we were given two datasets taken by UCI Machine Learning Repository. The goal of the exercise was to apply different pre-processing techniques and tune the hyperparameters of the machine learning algorithms, using the 10-fold cross validation method with grid search, in order to evaluate the performance of these algorithms on the dataset. 

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
