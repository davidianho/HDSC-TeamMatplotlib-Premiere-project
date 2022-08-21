# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import json
from sklearn.linear_model import LinearRegression


df = pd.read_csv('data.csv')
# Function to populate the list of countries in the dataset
# Also saving a copy of the list in a  JSON file which will be displayed if an incorrect country name is inputed by the user
def country_list_gen(df):
    df['Country Name'] = df['Country Name'].dropna().apply(lambda row: row.lower())
    lists = df['Country Name'].unique().tolist()
    with open('country_list.json','w', encoding='utf-8') as f:
        json.dump(lists, f, ensure_ascii=False,indent=4)
    return lists, df

# Function takes the country name and filters the dataframe, drops unnecessary fields, transpose and reset index of the dataframe.
def selecting_country(df,country):
    df = df.loc[(df['Country Name']==country) & (df['Series Code'] == 'SP.POP.TOTL')]
    df.drop(['Country Name','Country Code','Series Name','Series Code'],axis=1,inplace=True)
    df = df.T
    df.dropna(inplace=True)
    df = df.reset_index().rename(columns={'index':'year'})
    df['year'] = df['year'].apply(lambda x: x[:4])
    return df

def prediction_model(df):
    x = df.iloc[:, 0].values.reshape(-1,1)
    y = df.iloc[:, 1].values.reshape(-1,1)
    model = LinearRegression().fit(x,y)
    return model

def prediction(model, year):
    return int(model.coef_[0][0] * year + model.intercept_[0])
