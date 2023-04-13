# MLAustralianRainPrediction
Authors: Jessica van de Ven, Manon Le Donne, and Valeria Loria. 

## Motivation
This repository contains the work for a project in DS4400: Machine Learning and Data Mining 1 at Northeastern University. The goal is to take weather data from around Australia to predict when rain will occur on the next day. The variable to be predicted in the dataset is `RainTomorrow`.

## How to Use This Repo
1. Clone this repo using `git clone` in your CLI.
2. Launch Jupyter Lab with `jupyter lab` in your CLI.
3. Start with the `dataCleaningAndExplorations.ipynb` file, to clean and learn more about the dataset.
4. ... To be continued


## Abstract
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;This project aims to successfully predict whether it will rain tomorrow in various locations across Australia using several different classification models. The dataset required pre-processing to transform textual labels into categorical number representations and to calculate missing values in columns representing data such as temperature, and to remove rows with missing values in the target label observing next-day rain. Logistic Regression, Random Forest, and Feed-Forward Neural Network models are compared for measured accuracy and speed.

## Introduction
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Rain is a vital component of our natural ecosystem, and predicting when it will occur can be challenging. Our main goal for this project is to predict next-day rain by training several models on the target variable RainTomorrow. This dataset contains about 10 years of daily weather observations from many locations across Australia. We will compare our different models to determine which one does a better job at predicting the target variable. Ideally, they will be able to predict RainTomorrow with high accuracy.
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Climate change is a huge and controversial topic, and learning more about the climate around the world is important for further studies and discoveries. Predicting rain is just one aspect of this larger issue. By improving our ability to predict rain, we can gain insights into the changing climate and potentially help mitigate the effects of climate change. Additionally, our ability to predict rain has important implications for agriculture, water management, and disaster preparedness.
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;To tackle this problem, we are using three different approaches: Linear Regression, Random Forest Classifier, and Neural Networks. Each approach has its advantages and disadvantages, and we will compare their performance to determine which one works best for this problem. Linear regression is a simple approach that can give us a good baseline prediction. The Random Forest Classifier is a popular machine learning algorithm that can be used for both classification and regression tasks. Neural networks, on the other hand, can be more complex but have the potential to achieve higher accuracy.
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Compared to other competing methods, we believe that our approach is a good one for several reasons. First, we have access to a large and diverse dataset of daily weather observations from multiple locations across Australia. Second, we are using three different approaches, which will allow us to compare their performance and determine which one works best. Third, we are using machine learning algorithms that have been shown to work well for similar tasks in the past.
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The key components of our approach include data preprocessing, model training, and model evaluation. We will start by cleaning and preprocessing the dataset to ensure that it is ready for use with our machine learning algorithms. Then we will train our models using the different approaches we outlined earlier. Finally, we will evaluate their performance and determine which one works best for this problem. One potential limitation of our approach is that we are only using three different models. There may be other machine learning algorithms that could perform better, and we may not be able to identify them using our current approach. Another possible limitation pertains to the use of Neural Networks which work best with image data. To tackle this, we had to modify our data so that it could accommodate the model. 
## Setup
### Data Overview
This time series data was collected daily from 2008-12-01 to 2017-06-24. The dataset has 145,460 rows and 22 columns, described in Table 1. There are 31,877 rows with the RainTomorrow feature as true (21.9%), 110,316 rows with the RainTomorrow feature as false (75.8%), and 3,267 rows with the RainTomorrow value missing (2.2%). 

Table 1: Features in Dataset
| Column        | Type            | Count  | Mean     | Std    | Min   | Max    |
| ------------- | --------------- | ------ | -------- | ------ | ----- | ------ |
| Location      | enum string\*   | 145460 | N/A      | N/A    | N/A   | N/A    |
| MinTemp       | float64         | 143975 | 12.194   | 6.398  | \-8.5 | 33.9   |
| MaxTemp       | float64         | 144199 | 23.221   | 7.119  | \-4.8 | 48.1   |
| Rainfall      | float64         | 142199 | 2.360    | 8.478  | 0     | 371    |
| Evaporation   | float64         | 82670  | 5.468    | 4.193  | 0     | 145    |
| Sunshine      | float64         | 75625  | 7.611    | 3.785  | 0     | 14.5   |
| WindGustDir   | enum string\*\* | 135134 | N/A      | N/A    | N/A   | N/A    |
| WindDir9am    | enum string\*\* | 134894 | N/A      | N/A    | N/A   | N/A    |
| WindDir3pm    | enum string\*\* | 141232 | N/A      | N/A    | N/A   | N/A    |
| WindGustSpeed | float64         | 135197 | 40.035   | 13.607 | 6     | 135    |
| WindSpeed9am  | float64         | 143693 | 14.043   | 8.915  | 0     | 130    |
| WindSpeed3pm  | float64         | 142398 | 18.662   | 8.809  | 0     | 87     |
| Humidity9am   | float64         | 142806 | 68.880   | 19.029 | 0     | 100    |
| Humidity3pm   | float64         | 140953 | 51.539   | 20.795 | 0     | 100    |
| Pressure9am   | float64         | 130395 | 1017.65  | 7.106  | 980.5 | 1041   |
| Pressure3pm   | float64         | 130432 | 1015.256 | 7.037  | 977.1 | 1039.6 |
| Cloud9am      | float64         | 89572  | 4.447    | 2.887  | 0     | 9      |
| Cloud3pm      | float64         | 86102  | 4.509    | 2.720  | 0     | 9      |
| Temp9am       | float64         | 143693 | 16.990   | 6.488  | \-7.2 | 40.2   |
| Temp3pm       | float64         | 141851 | 21.683   | 6.936  | \-5.4 | 46.7   |
| RainToday     | binary string   | 142199 | N/A      | N/A    | N/A   | N/A    |
| RainTomorrow  | binary string   | 142193 | N/A      | N/A    | N/A   | N/A    |

\*The enumerated Location values: 'Albury', 'BadgerysCreek', 'Cobar', 'CoffsHarbour', 'Moree', 'Newcastle', 'NorahHead', 'NorfolkIsland', 'Penrith', 'Richmond', 'Sydney', 'SydneyAirport', 'WaggaWagga', 'Williamtown', 'Wollongong', 'Canberra', 'Tuggeranong', 'MountGinini', 'Ballarat', 'Bendigo', 'Sale', 'MelbourneAirport', 'Melbourne', 'Mildura', 'Nhil', 'Portland', 'Watsonia', 'Dartmoor', 'Brisbane', 'Cairns', 'GoldCoast', 'Townsville', 'Adelaide', 'MountGambier', 'Nuriootpa', 'Woomera', 'Albany', 'Witchcliffe', 'PearceRAAF', 'PerthAirport', 'Perth', 'SalmonGums', 'Walpole', 'Hobart', 'Launceston', 'AliceSprings', 'Darwin', 'Katherine', 'Uluru'
\*\*The enumerated WindGustDir, WindDir9am, and WindDir3pm values: ‘W', 'WNW', 'WSW', 'NE', 'NNW', 'N', 'NNE', 'SW', nan, 'ENE', 'SSE', 'S', 'NW', 'SE', 'ESE', 'E', 'SSW'

### Logistic Regression
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Before running and training the data using Scikit Learn’s library of logistic regression, the cleaned dataset needed to again be refined in order to fit it for a logistic regression. The cleaned dataset has a mixture of numerical and categorical features. In order to have a stable and more accurate model, it is necessary to scale or normalize the numerical values as logistic regressions are sensitive to the scale of the imputed values. So, larger scales can often dominate and create a biased or incorrect model. Scikit Learn’s library also includes a function of preprocessing called the MinMaxScalar that can be used to scale entire dataframes. The MinMaxScalar will transform columns proportionally within the range [0,1]. Doing this, instead of about the mean 0 with standard deviation of 1,  will not distort the data when transforming and preserve the shape of the dataset. Next, concerning categorical values it was also necessary to transform these values into dummy variables as logistic regression models are based on continuous variables and categorical values may not also reflect that. We represent the categorical values as binary numbers with 1 indicating the presence of the variable and 0 representing the absence of it. Now that the categorical values are all dummy variables we can more clearly see the effect of a categorical value on the outcome variable while also controlling the effects of the rest of the variables. Creating dummy variables also avoids issues of the model treating categorical variables as numerical which can lead to misleading coefficient estimates and predicted probabilities and so that we can create a more accurate relationship between feature variables and the outcome variable. 
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;When running the logistic regression, we used Python 3 in Jupyter Labs launched locally and since the dataset is large enough, we split the data into 20% testing and 80% training data. Then we fit the model using the Scikit Learn’s library and tested it. After which, we obtained an accuracy score, mean squared error score and a R^2 score.

### Random Forest Classifier
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The first step to implement the Random Forest Classifier is to import the necessary libraries, including pandas for data manipulation, scikit-learn's train_test_split function for splitting data into training and testing sets, RandomForestClassifier for creating a random forest model, and various metrics functions for evaluating the model's performance.
Next, the code reads in a cleaned dataset of weather data, which includes a date column that is parsed and set as the index column. The samples with missing target values are dropped from the dataset. The missing feature values in the remaining data are imputed using the SimpleImputer function from scikit-learn, which replaces missing values with the mean of the remaining values. The imputed data is saved as a new dataframe.
The data is then split into training and testing sets using the train_test_split function. The training data consists of 80% of the data, and the remaining 20% is used for testing. The 'RainTomorrow' column is dropped from the feature variables, and is saved as the target variable.
A random forest classifier is then initialized with hyperparameters, including 100 decision trees and a maximum depth of 5. The classifier is trained on the training data using the fit method. The predict method is then used to generate predictions on the testing data.

### Feed Forward Neural Network
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;To set up the neural network, libraries such as torch, pandas, numpy, and torchvision need to be imported. The torch library is used to create the structure of the neural network. The structure of the neural network has been customized to have multiple hidden layers, an activation function (ReLU, Tanh, Sigmoid), and applies the softmax function to output a classification result to one of the two output layers (predicting next-day rain, and predicting no next-day rain). During training, the loss is calculated using cross entropy loss. Hyperparameters such as the learning rate and number of epochs are specified. The data, after cleaning, is split into 80% training/20% testing and then processed into an iterative data loader. 


## Results
### Random Forest Classifier 
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;After analyzing the Random Forest Classifier model, we obtained the following results. The accuracy score of the model was 0.8177, which was computed using the accuracy_score function by comparing the predicted values to the actual values. Our accuracy score indicates that the model correctly classified 81.77% of the cases.The confusion matrix, which shows the number of true positive, false positive, true negative, and false negative values, was also calculated, and the results were as follows: [[21416 596 0], [ 4198 2219 3], [ 454 53 153]]. 
Moreover, the precision of the model, which measures the ratio of true positive predictions to the total predicted positives, was 0.8146. In other words, precision tells us how many of the predicted positive cases were actually positive. In our case, this means that out of all the cases predicted as positive, 81.46% were actually positive.The recall, which measures the ability of the model to correctly identify the positive cases, was the same as the accuracy score at 0.8177. The F1 Score metric is the harmonic mean of precision and recall. It provides a balance between precision and recall and is useful when the data is imbalanced. In our case, the F1 score was 0.7879, which indicates that the model achieved a good balance between precision and recall. 
Lastly, we computed the AUC-ROC score using the roc_auc_score function, which measures the model's ability to correctly rank the probabilities of the positive class. The micro averaging method was used for multiclass classification, and the AUC-ROC score (micro) was 0.9432. This score indicates that the model has a high ability to distinguish between the positive and negative cases.

### Feed Forward Neural Network
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Initially, the accuracy of this model is 0.773 and after some tuning of the structure and normalizing the data, the accuracy is 0.776, so there are current concerns about overfitting. 

## Discussion
### Random Forest Classifier 
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Based on the results obtained above, the Random Forest Classifier model achieved a good performance. The confusion matrix shows that the model correctly identified a large number of true positives (21416), while the false positives (596) and false negatives (4198) were relatively low. The results indicate that the model has a high ability to distinguish between the positive and negative cases. To compare these results with existing approaches, we searched the Kaggle dataset page for similar approaches.Based on the discussions, it appears that the results obtained by the Random Forest Classifier model are competitive with other approaches. Overall, the results obtained by the Random Forest Classifier model are promising, and the model appears to be a viable option for this classification task. 



## Conclusion
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;In conclusion, this project was an experiment using machine learning techniques that, if it were possible to predict if it would rain the next day in Australia. Fueled by the imminence of the climate crisis we wanted to use these machine learning approaches in hopes of creating an informative tool that would help predict rain throughout Australia. Using the three approaches of logistic regression, random forest classifier, and Neural networks we found our results to be….

## Sources
["Rain in Australia" Kaggle Dataset](https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package?resource=download)