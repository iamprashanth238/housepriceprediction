from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import pandas as pd

data = pd.read_csv('D:\Datasets\melb_data.csv')

data = data.dropna()

y = data.Price

features = ['Rooms','Bathroom','Landsize','Lattitude','Longtitude']

X = data[features]

# building model with DecisionTreeRegressor
dt_model = DecisionTreeRegressor(random_state=1)

#fitting model with actual data
dt_model.fit(X,y)
dt_predic1 = dt_model.predict(X)
mae1 = mean_absolute_error(dt_predic1,y)
print("Decision Tree Regressor")
print("Mean Absolute Error for actual data:")
print(mae1)

#now split the data into train_X, val_x, train_y, val_y
train_X, val_X, train_y, val_y = train_test_split(X,y, random_state=1)

# fitting the model with split data for DesicionTreeRegressor
dt_model.fit(train_X,train_y)
dt_predic2 = dt_model.predict(val_X)
mae2 = mean_absolute_error(dt_predic2,val_y)
print("Mean Absolute Error for split data:")
print(mae2)


# lets built model with another method with RandomForestRegressor
rf_model = RandomForestRegressor(random_state = 1)

rf_model.fit(X,y)
rf_predic1 = rf_model.predict(X)

# mean absolute error
mae3 = mean_absolute_error(rf_predic1, y)
print("\nRandomForestRegressor")
print("Mean Absoulte Error for Raw data : ")
print(mae3)

#lets fit with split data 
rf_model.fit(train_X,train_y)
rf_predic2 = rf_model.predict(val_X)

# absolute error
mae4 = mean_absolute_error(rf_predic2, val_y)
print("Mean Absoulte Error for split data : ")
print(mae4)