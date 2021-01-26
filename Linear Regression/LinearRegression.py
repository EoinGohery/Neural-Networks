#Imports
import pandas as pd
import numpy as np
import keras
import seaborn as sns
from keras.models import Sequential
from keras.layers import Dense
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#Read data from csv stored on Github
rawData = pd.read_csv('https://raw.githubusercontent.com/EoinGohery/LinearRegression/main/kc_house_data.csv', engine='python', error_bad_lines=False, sep=',')
rawData.head()

#Remove unwanted values
rawData = rawData.drop(columns=["id","date","zipcode","lat","long","yr_renovated","sqft_living15", "sqft_lot15"])
rawData

#Visualise rawData
sns.pairplot(rawData[['price', 'sqft_lot' , 'sqft_living', 'bedrooms']], diag_kind='kde')

#Prepare the input X and Output Y values
X = rawData.iloc[:,0:12].values
Y = rawData.iloc[:,0].values

#Normalise the input values within 0-1 
min_max_scaler = preprocessing.MinMaxScaler()
X_scale = min_max_scaler.fit_transform(X)
X_scale

#Remove comment from next line to use output normalisation
#Y = np.interp(Y, (Y.min(), Y.max()), (0, +1))

#Split the values into Trainind Data, Test Data and Validation Data
X_train, X_val_and_test, Y_train, Y_val_and_test = train_test_split(X_scale, Y, test_size=0.3)
X_val, X_test, Y_val, Y_test = train_test_split(X_val_and_test, Y_val_and_test, test_size=0.5)
print(X_train.shape, X_val.shape, X_test.shape, Y_train.shape, Y_val.shape, Y_test.shape)

#Model Setup
model = Sequential([
    Dense(64, activation='relu', input_shape=(12,)),
    Dense(32, activation='relu'),
    Dense(1, activation='linear'),
])
model.compile(optimizer="adam",loss='mean_squared_logarithmic_error', metrics=['mse'])

#Run Training 
hist = model.fit(X_train, Y_train, batch_size=64, epochs=50, validation_data=(X_val, Y_val), verbose=1)

#Plot Loss Graph
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.show()

#Plot Alternate Loss Graph (MSE)
plt.title('MSE Loss')
plt.plot(hist.history['mse'])
plt.plot(hist.history['val_mse'])
plt.xlabel('Epoch')
plt.legend(['MSE', 'MAE'], loc='upper right')
plt.show()