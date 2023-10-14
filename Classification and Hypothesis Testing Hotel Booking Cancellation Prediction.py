#!/usr/bin/env python
# coding: utf-8

# In[1]:


str1="helloworld" 
str1[:-1] 


# In[2]:


# Importing the basic libraries we will require for the project

# Libraries to help with reading and manipulating data
import pandas as pd
import numpy as np

# Libaries to help with data visualization
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# Importing the Machine Learning models we require from Scikit-Learn
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

# Importing the other functions we may require from Scikit-Learn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OneHotEncoder

# To get diferent metric scores
from sklearn.metrics import confusion_matrix,classification_report,roc_auc_score,plot_confusion_matrix,precision_recall_curve,roc_curve,make_scorer

# Code to ignore warnings from function usage
import warnings;
import numpy as np
warnings.filterwarnings('ignore')


# In[ ]:


hotel = pd.read_csv("INNHotelsGroup.csv")


# In[ ]:


# Copying data to another variable to avoid any changes to original data
data = hotel.copy()


# In[ ]:


data.head()


# In[ ]:


data.tail()


# In[ ]:


data.shape


# In[ ]:


#Check the data types of the columns for the dataset
data.info()


# In[ ]:


#Dropping duplicate values
data.duplicated().sum()


# In[ ]:


#Dropping the unique values column Booking_ID first before we proceed forward, as a column with unique values will have almost no predictive power for the Machine Learning problem at hand.

data = data.drop(["Booking_ID"], axis=1)


# In[ ]:


data.head()


# In[ ]:


#check summary statistics of the data
data.describe()


# In[ ]:


#Exploratory Data Analysis : Univariate Analysis
#Let's explore these variables in some more depth by observing their distributions. 
#We will first define a hist_box() function that provides both a boxplot and a histogram in the same visual, with which we can perform univariate analysis on the columns of this dataset.

# Defining the hist_box() function
def hist_box(data,col):
  f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={'height_ratios': (0.15, 0.85)}, figsize=(12,6))
  # Adding a graph in each part
  sns.boxplot(data[col], ax=ax_box, showmeans=True)
  sns.distplot(data[col], ax=ax_hist)
  plt.show()


# In[ ]:


#plotting a histogram and box plot for the variable lead time 
hist_box(data,"lead_time") 


# In[ ]:


#plotting the histogram and box plot for Average Price per Room
hist_box(data,"avg_price_per_room")


# In[ ]:


#checking which rooms are priced equal to 0
data[data["avg_price_per_room"] == 0]


# In[ ]:


#checking the number of rooms priced equal to 0 per market segment
data.loc[data["avg_price_per_room"] == 0, "market_segment_type"].value_counts()


# In[ ]:


# Calculating the 25th quantile
Q1 = data["avg_price_per_room"].quantile(0.25)

# Calculating the 75th quantile
Q3 = data["avg_price_per_room"].quantile(0.75)

# Calculating IQR
IQR = Q3 - Q1

# Calculating value of upper whisker
Upper_Whisker = Q3 + 1.5 * IQR
Upper_Whisker


# In[ ]:


# assigning the outliers the value of upper whisker
data.loc[data["avg_price_per_room"] >= 500, "avg_price_per_room"] = Upper_Whisker


# In[ ]:


#Let's understand the distribution of the categorical variables: Number of Children

sns.countplot(data['no_of_children'])
plt.show()


# In[ ]:


data['no_of_children'].value_counts(normalize=True)


# In[ ]:


# replacing 9, and 10 children with 3
data["no_of_children"] = data["no_of_children"].replace([9, 10], 3)


# In[ ]:


#Let's understand the distribution of the categorical variables: Arrival month

sns.countplot(data["arrival_month"])
plt.show()


# In[ ]:


data['arrival_month'].value_counts(normalize=True)


# In[3]:


#Let's understand the distribution of the categorical variables: Booking status
sns.countplot(data["booking_status"])
plt.show()


# In[ ]:


data['booking_status'].value_counts(normalize=True)


# In[ ]:


#encoding Canceled bookings to 1 and Not_Canceled as 0 for further analysis

data["booking_status"] = data["booking_status"].apply(
    lambda x: 1 if x == "Canceled" else 0
)


# In[4]:


#Bivariate Analysis 
# Visualizing the correlation matrix using a heatmap

cols_list = data.select_dtypes(include=np.number).columns.tolist()

plt.figure(figsize=(12, 7))
sns.heatmap(data[cols_list].corr(), annot=True, vmin=-1, vmax=1, fmt=".2f", cmap="Spectral")
plt.show()


# In[ ]:


#Noting that Hotel rates are dynamic and change according to demand and customer demographics. Let's see how prices vary across different market segments
plt.figure(figsize=(10, 6))
sns.boxplot(
    data=data, x="market_segment_type", y="avg_price_per_room", palette="gist_rainbow"
)
plt.show()


# In[ ]:


# We will define a stacked barplot() function to help analyse how the target variable varies across predictor categories.
# Defining the stacked_barplot() function
def stacked_barplot(data,predictor,target,figsize=(10,6)):
  (pd.crosstab(data[predictor],data[target],normalize='index')*100).plot(kind='bar',figsize=figsize,stacked=True)
  plt.legend(loc="lower right")
  plt.ylabel('Percentage Cancellations %')


# In[ ]:


#Plotting a stacked barplot for the variable Market Segment Type against the target variable Booking Status
stacked_barplot(data, "market_segment_type", "booking_status" )


# In[ ]:


# Repeating guests are the guests who stay in the hotel often and are important to brand equity.
# Plot the stacked barplot for the variable Repeated Guest against the target variable Booking Status
stacked_barplot(data, "repeated_guest", "booking_status" )


# In[ ]:


# Let's analyze the customer who stayed for at least a day at the hotel.

stay_data = data[(data["no_of_week_nights"] > 0) & (data["no_of_weekend_nights"] > 0)]
stay_data["total_days"] = (stay_data["no_of_week_nights"] + stay_data["no_of_weekend_nights"])

stacked_barplot(stay_data, "total_days", "booking_status",figsize=(15,6))


# In[5]:


# As hotel room prices are dynamic, Let's see how the prices vary across different months
plt.figure(figsize=(10, 5))
sns.lineplot(y=data["avg_price_per_room"], x=data["arrival_month"], ci=None)
plt.show()


# In[ ]:


# Data Preparation for Modeling
# We want to predict which bookings will be canceled. Before we proceed to build a model, we'll have to encode categorical features.We'll split the data into train and test to be able to evaluate the model that we build on the train data.

# Separating the independent variables (X) and the dependent variable (Y)

X = data.drop(["booking_status"], axis=1)
Y = data["booking_status"]

X = pd.get_dummies(X, drop_first=True) # Encoding the Categorical features


# In[6]:


# Splitting the data into a 70% train and 30% test set

# Splitting data in train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30,stratify=Y, random_state=1)

#we have used stratified sampling technique to ensure that relative class frequencies are approximately preserved in each train and validation fold in order to avoid large imbalance in the distribution of the target classes: for instance there could be several times more negative samples than positive samples.


# In[ ]:


print("Shape of Training set : ", X_train.shape)
print("Shape of test set : ", X_test.shape)
print("Percentage of classes in training set:")
print(y_train.value_counts(normalize=True))
print("Percentage of classes in test set:")
print(y_test.value_counts(normalize=True))


# In[ ]:


# To scale the data using z-score 
from sklearn.preprocessing import StandardScaler

# Scaling the data
sc=StandardScaler()

# Fit_transform on train data
X_train_scaled=sc.fit_transform(X_train)
X_train_scaled=pd.DataFrame(X_train_scaled, columns=X.columns)

# Transform on test data
X_test_scaled=sc.transform(X_test)
X_test_scaled=pd.DataFrame(X_test_scaled, columns=X.columns)


# In[ ]:


# Model Evaluation Criterion : The hotel would want the F1 Score to be maximized, the greater the F1 score, the higher the chances of minimizing False Negatives and False Positives.
# let's create a function to calculate and print the classification report and confusion matrix so that we don't have to rewrite the same code repeatedly for each model.

# Creating metric function 
def metrics_score(actual, predicted):
    print(classification_report(actual, predicted))

    cm = confusion_matrix(actual, predicted)
    plt.figure(figsize=(8,5))
    
    sns.heatmap(cm, annot=True,  fmt='.2f', xticklabels=['Not Cancelled', 'Cancelled'], yticklabels=['Not Cancelled', 'Cancelled'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()


# In[ ]:


#Training the model using mode of target
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
pred_test = []
for i in range (0, 10883):
    pred_test.append(y_train.mode()[0])

#Printing f1 and accuracy scores    
print('The accuracy for mode model is:', accuracy_score(y_test, pred_test))
print('The f1 score for the model model is:',f1_score(y_test, pred_test))

#Ploting the cunfusion matrix
confusion_matrix(y_test, pred_test)


# In[ ]:


# Building the model
# We will be building 4 different models:Logistic Regression, Support Vector Machine (SVM), Decision Tree and Random Forest

#Logistic Regression

# instantiate the model
lg = LogisticRegression()

# fit the model
lg.fit(X_train, y_train)  


# In[ ]:


# Checking the performance on the training data
y_pred_train = lg.predict(X_train)
metrics_score(y_train, y_pred_train)


# In[ ]:


# Checking the performance on the test dataset
y_pred_test = lg.predict(X_test)
metrics_score(y_test, y_pred_test)


# In[ ]:


# Finding the optimal threshold for the model using the Precision-Recall Curve


# Predict_proba gives the probability of each observation belonging to each class
y_scores_lg=lg.predict_proba(X_train)
precisions_lg, recalls_lg, thresholds_lg = precision_recall_curve(y_train, y_scores_lg[:,1])

# Plot values of precisions, recalls, and thresholds
plt.figure(figsize=(10,7))
plt.plot(thresholds_lg, precisions_lg[:-1], 'b--', label='precision')
plt.plot(thresholds_lg, recalls_lg[:-1], 'g--', label = 'recall')
plt.xlabel('Threshold')
plt.legend(loc='upper left')
plt.ylim([0,1])
plt.show()


# In[ ]:


# Setting the optimal threshold
optimal_threshold = 0.42


# In[ ]:


#  Check the performance of the model on train and test data using the optimal threshold
# creating confusion matrix
y_pred_train = lg.predict_proba(X_train)
metrics_score(y_train, y_pred_train[:,1]>optimal_threshold)


# In[ ]:


# Let's check the performance on the test set
y_pred_test = lg.predict_proba(X_test)
metrics_score(y_test, y_pred_test[:,1]>optimal_threshold)


# In[ ]:


# Support Vector Machines
# To accelerate SVM training, let's scale the data for support vector machines.

scaling = MinMaxScaler(feature_range=(-1,1)).fit(X_train)
X_train_scaled = scaling.transform(X_train)
X_test_scaled = scaling.transform(X_test)


# In[ ]:


# Let's build the models using the two of the widely used kernel functions: Linear Kernel and RBF Kernel

# Build a Support Vector Machine model using a linear kernel
# Please use the scaled data for modeling Support Vector Machine

svm = SVC(kernel='linear',probability=True) # Linear kernal or linear decision boundary
model = svm.fit(X = X_train_scaled, y = y_train)


# In[ ]:


# Check the performance of the model on train and test data
y_pred_train_svm = model.predict(X_train_scaled)
metrics_score(y_train, y_pred_train_svm)


# In[ ]:


# Checking model performance on test set

print("Testing performance:")
y_pred_test_svm = model.predict(X_test_scaled)
metrics_score(y_test, y_pred_test_svm)


# In[ ]:


# Find the optimal threshold for the model using the Precision-Recall Curve

# Predict on train data
y_scores_svm=model.predict_proba(X_train_scaled)

precisions_svm, recalls_svm, thresholds_svm = precision_recall_curve(y_train, y_scores_svm[:,1])

# Plot values of precisions, recalls, and thresholds
plt.figure(figsize=(10,7))
plt.plot(thresholds_svm, precisions_svm[:-1], 'b--', label='precision')
plt.plot(thresholds_svm, recalls_svm[:-1], 'g--', label = 'recall')
plt.xlabel('Threshold')
plt.legend(loc='upper left')
plt.ylim([0,1])
plt.show()


# In[ ]:


optimal_threshold_svm= 0.41


# In[ ]:


# Check the performance of the model on train and test data using the optimal threshold
print("Training performance:")
y_pred_train_svm = model.predict_proba(X_train_scaled)
metrics_score(y_train, y_pred_train_svm[:,1]>optimal_threshold_svm)


# In[ ]:


y_pred_test = model.predict_proba(X_test_scaled)
metrics_score(y_test, y_pred_test[:,1]>optimal_threshold_svm)


# In[ ]:


# Build a Support Vector Machines model using an RBF kernel

svm_rbf=SVC(kernel='rbf',probability=True)
# Fit the model
svm_rbf.fit(X_train_scaled,y_train)


# In[ ]:


#  Check the performance of the model on train and test data
y_pred_train_svm = svm_rbf.predict(X_train_scaled)
metrics_score(y_train, y_pred_train_svm)


# In[ ]:


# Checking model performance on test set
y_pred_test = svm_rbf.predict(X_test_scaled)
metrics_score(y_test, y_pred_test)


# In[ ]:


# Predict on train data
y_scores_svm=svm_rbf.predict_proba(X_train_scaled)

precisions_svm, recalls_svm, thresholds_svm = precision_recall_curve(y_train, y_scores_svm[:,1])

# Plot values of precisions, recalls, and thresholds
plt.figure(figsize=(10,7))
plt.plot(thresholds_svm, precisions_svm[:-1], 'b--', label='precision')
plt.plot(thresholds_svm, recalls_svm[:-1], 'g--', label = 'recall')
plt.xlabel('Threshold')
plt.legend(loc='upper left')
plt.ylim([0,1])
plt.show()


# In[ ]:


optimal_threshold_svm= 0.40


# In[ ]:


# Check the performance of the model on train and test data using the optimal threshold
y_pred_train_svm = model.predict_proba(X_train_scaled)
metrics_score(y_train, y_pred_train_svm[:,1]>optimal_threshold_svm)


# In[ ]:


metrics_score(y_test, y_pred_test[:,1]>optimal_threshold_svm)


# In[ ]:


# Build a Decision Tree Model

model_dt = DecisionTreeClassifier(random_state=1)
model_dt.fit(X_train, y_train)


# In[ ]:


# Checking performance on the training dataset
pred_train_dt = model_dt.predict(X_train)
metrics_score(y_train, pred_train_dt)


# In[ ]:


# Checking model performance on test set
pred_test_dt = model_dt.predict(X_test)
metrics_score(y_test, pred_test_dt)


# In[ ]:


# Perform hyperparameter tuning for the decision tree model using GridSearch CV
# Choose the type of classifier.
estimator = DecisionTreeClassifier(random_state=1)

# Grid of parameters to choose from
parameters = {
    "max_depth": np.arange(2, 7, 2),
    "max_leaf_nodes": [50, 75, 150, 250],
    "min_samples_split": [10, 30, 50, 70],
}


# Run the grid search
grid_obj = GridSearchCV(estimator, parameters, cv=5,scoring='recall',n_jobs=-1)
grid_obj = grid_obj.fit(X_train, y_train)

# Set the clf to the best combination of parameters
estimator = grid_obj.best_estimator_

# Fit the best algorithm to the data.
estimator.fit(X_train, y_train)


# In[ ]:


# Checking performance on the training dataset
dt_tuned = estimator.predict(X_train)
metrics_score(y_train,dt_tuned)


# In[ ]:


# Checking performance on the test dataset
y_pred_tuned = estimator.predict(X_test)
metrics_score(y_test,y_pred_tuned)


# In[ ]:


# Visualizing the Decision Tree

feature_names = list(X_train.columns)
plt.figure(figsize=(20, 10))
out = tree.plot_tree(
    estimator,max_depth=3,
    feature_names=feature_names,
    filled=True,
    fontsize=9,
    node_ids=False,
    class_names=None,
)
# below code will add arrows to the decision tree split if they are missing
for o in out:
    arrow = o.arrow_patch
    if arrow is not None:
        arrow.set_edgecolor("black")
        arrow.set_linewidth(1)
plt.show()


# In[ ]:


# Importance of features in the tree building

importances = estimator.feature_importances_
indices = np.argsort(importances)

plt.figure(figsize=(8, 8))
plt.title("Feature Importances")
plt.barh(range(len(indices)), importances[indices], color="violet", align="center")
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel("Relative Importance")
plt.show()


# In[ ]:


# Build a Random Forest Model
rf_estimator = RandomForestClassifier( random_state = 1)

rf_estimator.fit(X_train, y_train)


# In[ ]:


# Check the performance of the model on the train data
y_pred_train_rf = rf_estimator.predict(X_train)

metrics_score(y_train, y_pred_train_rf)


# In[ ]:


# Check the performance of the model on the test data
y_pred_test_rf = rf_estimator.predict(X_test)

metrics_score(y_test, y_pred_test_rf)


# In[7]:


# Let's check the feature importance of the Random Forest

importances = rf_estimator.feature_importances_

columns = X.columns

importance_df = pd.DataFrame(importances, index = columns, columns = ['Importance']).sort_values(by = 'Importance', ascending = False)

plt.figure(figsize = (13, 13))

sns.barplot(importance_df.Importance, importance_df.index)


# In[ ]:


#Offer insights on your analysis and prediction in order to reduce cancellations

