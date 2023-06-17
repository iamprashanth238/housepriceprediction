import pandas as pd
import numpy as np

#reading the daa from the dataset
data = pd.read_csv('C:/Users/Admin/Desktop/melb_data.csv')

#analize the data by describing
#print(data.describe())

#removing the missing values using dropna()
data = data.dropna()

#first know what are are columns are there using columns
name_columns = data.columns
#print(name_columns)

# targeting a output variable i.e in my situation prices and storing in y
y = data.Price
#print(y.head())

# above y is output. But we need to give input for the for model
# there are somany variable/columns we need only some columns which helps to find the output
# output is depend on the features we give 

#selecting few columns
features = ['Rooms','Bathroom','Landsize','Lattitude','Longtitude']

#now including this features to the existing data set and storing in the X
X = data[features]

# now we perform regression between the X,y
# for that we are using sklearn.tree using DecisionTreeRegressor
from sklearn.tree import DecisionTreeRegressor

#now create a model 
home_model = DecisionTreeRegressor(random_state=1)

#now fit the X,y to the model
home_model.fit(X,y)

#now we predict the output of first five houes
print("The price prediction of 5 houses are :")
print(X.head())
print("Price prediction are :")
print(home_model.predict(X.head()))

# wow we created a simple model