import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle


data = pd.read_csv(r"C:\Data Science\Project Assignments and Project Topic list\Car Price Prediction\processes2.csv")

data = data[['year','km_driven','fuel','seller_type','transmission','owner','seats','selling_price']]


data = pd.get_dummies(data)


X = data.drop('selling_price', axis=1)
y = data['selling_price']


model = LinearRegression()
model.fit(X, y)

pickle.dump(model, open("model.pkl", "wb"))


pickle.dump(X.columns, open("columns.pkl", "wb"))

print("Model trained successfully")