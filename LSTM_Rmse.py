# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 09:07:00 2022

@author: a454g185
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 11:59:18 2022
@author: a454g185
"""

from pandas import datetime
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, SimpleRNN, GRU
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from sklearn.impute import KNNImputer
import datetime
import tensorflow as tf

np.random.seed(1234)
tf.random.set_seed(1234)


""" 
    Reading dataset from github repo
"""
series = pd.read_csv(r'https://raw.githubusercontent.com/armangh67/LSTM/main/Turbine_Data.csv')

"""
    Pre-processing the dataset
"""
# print(series.columns)
df_selected = series[['Unnamed: 0','ActivePower','AmbientTemperatue','WindDirection','WindSpeed']]
df = df_selected.rename(columns = {'Unnamed: 0':'Date'})     ####### rename the date column ######


############## KNN for null values ################
def fill_null(data,n):
    knn = KNNImputer(n_neighbors = n, add_indicator = True)
    knn.fit(data)
    df = pd.DataFrame(knn.transform(data))
    a = list(data)
    x = []
    for i in range(data.shape[1]):
        x.append(np.array(df.iloc[:,i]))
    for i in range(data.shape[1]):
        data[(a[i])] = x[i] 
    return data



df['Date'] = pd.to_datetime(df['Date'])
dataset = df.set_index('Date')
dataset = dataset.loc['2020-01-01':]
dataset = fill_null(dataset, 50)     ####### Null value imputation ######


############ Select features for the modeling ############

# features_final = ['ActivePower','WindSpeed']
features_final = ['ActivePower','AmbientTemperatue','WindDirection','WindSpeed']
dataset = dataset[features_final]

"""
Defining functions: 1- Normalize function
                    2- Converting Data to Time series format
                    3- Train-Test-Split function
                    4- LSTM Model
"""
#################### function 1 : Normalizing data between 0 and 1 ############
def normalize(data, feature_range):   
    #normalize the dataset for working with the lstm nn
    scaler = MinMaxScaler(feature_range)
    data_normd = scaler.fit_transform(data)
    
    return data_normd,scaler


######################## function 2 : Convert dataset to time series format ###################

def Time_Series(ts: np.array, lag=1, n_ahead=1, target_index=0) -> tuple:
    """
    A method to create X and Y matrix from a time series array for the training of 
    deep learning models 
    """
    # Extracting the number of features that are passed from the array 
    n_features = ts.shape[1]
    
    # Creating placeholder lists
    X, Y = [], []

    if len(ts) - lag <= 0:
        X.append(ts)
    else:
        for i in range(len(ts) - lag - n_ahead):
            Y.append(ts[(i + lag):(i + lag + n_ahead), target_index])
            X.append(ts[i:(i + lag)])

    X, Y = np.array(X), np.array(Y)

    # Reshaping the X array to an LSTM input shape 
    X = np.reshape(X, (X.shape[0], lag, n_features))
    return X, Y

################### function 3 : Split data to train and test ###################
def Train_Test_Split(X,Y,ratio):
    Xtrain, Ytrain = X[0:int(X.shape[0] * train_share)], Y[0:int(X.shape[0] * train_share)]
    Xtest, Ytest = X[int(X.shape[0] * train_share):], Y[int(X.shape[0] * train_share):]
    return Xtrain, Ytrain, Xtest, Ytest


##################### function 4 : LSTM model ##############################
def LSTM_model(n_lag, n_pred, unit, dropout, lr):
    model = Sequential()
    model.add(LSTM(unit, activation = 'relu', input_shape = (n_lag , num_features) , return_sequences = True))
    model.add(Dropout(dropout))
    model.add(LSTM(unit , activation = 'tanh',  return_sequences = True))
    model.add(Dropout(dropout))
    model.add(LSTM(unit , activation = 'tanh', return_sequences = False))
    model.add(Dense(n_pred))    
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, mode = 'min')    
    model.compile(loss = tf.losses.MeanSquaredError(), optimizer = tf.keras.optimizers.Adam(learning_rate=lr), metrics=[tf.metrics.MeanAbsoluteError()])
    history = model.fit(Xtrain, Ytrain,epochs=epochs, batch_size = batch_size, validation_data=(Xtest, Ytest), shuffle=False, callbacks=[early_stopping])  ### for early stopping ####
    return model

def RNN_model(n_lag,n_pred,unit):
    model = Sequential()
    model.add(SimpleRNN(unit, activation='relu',input_shape = (n_lag,num_features)))
    # model.add(Dense(8, activation='tanh'))
    model.add(Dense(n_pred))
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, mode = 'min')    
    model.compile(loss = tf.losses.MeanSquaredError(), optimizer = tf.keras.optimizers.Adam(), metrics=[tf.metrics.MeanAbsoluteError()])
    history = model.fit(Xtrain, Ytrain,epochs=epochs, batch_size = batch_size, validation_data=(Xtest, Ytest), shuffle=False, callbacks=[early_stopping])  ### for early stopping ####
    return model

def GRU_model(n_lag, n_pred, unit, dropout, lr):
    model = Sequential()
    model.add(GRU(unit, activation = 'tanh', input_shape = (n_lag , num_features) , return_sequences = True))
    model.add(Dropout(dropout))
    model.add(GRU(unit , activation = 'tanh',  return_sequences = True))
    model.add(Dropout(dropout))
    model.add(GRU(unit , activation = 'tanh', return_sequences = False))
    model.add(Dense(n_pred))    
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, mode = 'min')    
    model.compile(loss = tf.losses.MeanSquaredError(), optimizer = tf.keras.optimizers.Adam(learning_rate=lr), metrics=[tf.metrics.MeanAbsoluteError()])
    history = model.fit(Xtrain, Ytrain,epochs=epochs, batch_size = batch_size, validation_data=(Xtest, Ytest), shuffle=False, callbacks=[early_stopping])  ### for early stopping ####
    return model

def CNN_model(n_lag, n_pred, unit):
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(n_lag, num_features)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(Xtrain, Ytrain, epochs=30)
    return model
    

"""
    Simulation Parameters
"""




 #### how many hours we want to predict ahead #####
n_hour_observation = 1     #### how many hours to observe #####   
lag = n_hour_observation*6
n_future = 6

train_share = 0.8 # ratio of observations for training from total series
epochs = 20
batch_size = 64
num_features = dataset.shape[1]
df_normd , scaler = normalize(dataset, feature_range=(0,1))
# Creating the X and Y for training, the formula is set up to assume the target Y is the left most column = target_index=0
# X, Y = Time_Series(df_normd, lag = lag, n_ahead = n_future)
# # Spliting into train and test sets 
# Xtrain, Ytrain, Xtest, Ytest = Train_Test_Split(X, Y, ratio = train_share)
# model = LSTM_model(n_lag=lag, n_pred=n_future, unit=64, dropout = 0.2, lr=0.001)
# yhat = model.predict(Xtest, verbose = 0)


def preds(Ypred,Yact):
    forecast = np.array(Ypred[0])
    actual = np.array(Yact[0])
    y_forecast = []
    y_act = []
    for i in range(Ypred.shape[0]-1):
        y_forecast.append(np.array(Ypred[i+1,0]))
        y_act.append(np.array(Yact[i+1,0]))
    y_forecast = np.concatenate([forecast,y_forecast])
    y_act = np.concatenate([actual,y_act])
    pred = pd.DataFrame(y_forecast)
    act = pd.DataFrame(y_act)
    pr_p = pd.concat([pred]*(num_features), axis=1)
    ac_p = pd.concat([act]*(num_features), axis=1)
    # inverse scale tranform the series back to kiloWatts of power
    pr_p = pd.DataFrame(scaler.inverse_transform(pr_p))
    ac_p = pd.DataFrame(scaler.inverse_transform(ac_p))
    #rename columns
    pr_p = pr_p.rename(columns={0:'PredPower'})
    ac_p = ac_p.rename(columns={0:'ActualPower'})
    df_final = pd.concat([ pr_p['PredPower'], ac_p['ActualPower']], axis=1)
    return df_final


win_length = 16

def RMSE_cal(model,window_size):
    rmse =[]
    for i in range(1,win_length):
        X, Y = Time_Series(df_normd, lag = lag, n_ahead = i)
        # Spliting into train and test sets 
        Xtrain, Ytrain, Xtest, Ytest = Train_Test_Split(X, Y, ratio = train_share)
        model = LSTM_model(n_lag=lag, n_pred=i, unit=64, dropout = 0.2, lr=0.001)
        yhat = model.predict(Xtest, verbose = 0)
        df_final = preds(yhat, Ytest)
        RMSE = sqrt(mean_squared_error(df_final['PredPower'],df_final['ActualPower'] ))
        rmse.append(RMSE)
        return rmse


# plt.plot(rmse)






# ##################### plot ###################

# df_final['hour']=range(1,len(df_final)+1)
# df_final['hour'] = df_final['hour'].astype(float)
# df_final['hour']*=1/6

# df_final = df_final.head(144)


# # for index in df_final.index:    
# #     a = df_final['PredPower'][index]
# #     if a < 0 :
# #         df_final['PredPower'][index] = 0

# #plot n_steps ahead for predicted and actual data
# plt.figure(figsize=(15, 8))
# plt.plot(df_final.hour, df_final.ActualPower, color='C0', marker='o', label='Actual Power')
# plt.plot(df_final.hour, df_final.PredPower, color='C1', marker='o', label='Predicted Power')
# plt.title('Predicted vs Actual Power')
# plt.xlabel("Time [Hour]", fontsize = 15)
# plt.ylabel("Power[kw]",fontsize = 15)
# plt.grid()
# plt.gcf().axes[0].yaxis.get_major_formatter().set_scientific(False)
# plt.legend()
# plt.savefig('forecast_example.png')
# plt.show