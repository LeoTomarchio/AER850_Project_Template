import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

"""OneHotEncoder"""
df = pd.read_csv('aer850_f24-main/data/housing.csv')
my_encoder = OneHotEncoder()
my_encoder.fit(df[['ocean_proximity']])
encoded_data = my_encoder.transform(df[['ocean_proximity']])
category_names = my_encoder.get_feature_names_out()
print(encoded_data)
encoded_data_df = pd.DataFrame(encoded_data, columns=category_names)
df = pd.concat([df,encoded_data_df],axis=1)
df = df.drop(columns=['ocean_proximity'])
df.to_csv("test2.csv")

"""Define X and Y"""
x_columns = ["longitude", "latitude", "housing_median_age", "total_rooms", "total_bedrooms", "population", "households", "median_income"]
y_column = ["median_house_value"]
X = df[x_columns]
y = df[y_column]

"""Train Test Split"""
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

"""Stratified Sampling Based on Income"""
df['median_income'].hist()


