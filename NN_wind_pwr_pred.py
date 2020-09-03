# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 13:18:43 2020

@author: AT

Wind Power prediction
"""

# # Import main packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

# # =========== 0: Loading Data =============
# Read the input data 
df = pd.read_csv('power_training.csv', index_col=0, encoding='utf-8')
df_test = pd.read_csv('power_test.csv', index_col=0, encoding='utf-8')

# Data set split already done
X_train = df[['Average wind speed (m/s)', 'Input turbulence intensity (%)', 'Input alpha']]
y_train = df['Average power (kW)']

X_test= df_test[['Average wind speed (m/s)', 'Input turbulence intensity (%)', 'Input alpha']] 
y_test= df_test['Average power (kW)'] 

# Visualizing The Data
df.plot.scatter(x='Average wind speed (m/s)', y='Average power (kW)')
df_test.plot.scatter(x='Average wind speed (m/s)', y='Average power (kW)')

# # =========== 1: Normalization/Standardization =============
#X_train_df = pd.read_csv('power_training.csv').drop(columns = ['Average power (kW)'])
#y_train_df = pd.read_csv('power_training.csv')['Average power (kW)']
#X_test_df = pd.read_csv('power_test.csv').drop(columns = ['Average power (kW)'])
#y_test_df = pd.read_csv('power_test.csv')['Average power (kW)']

# Standardize features
# from sklearn.preprocessing import StandardScaler
#scaler = StandardScaler().fit(X_train_df)
#X_train = scaler.transform(X_train_df)
#X_test = scaler.transform(X_test_df)
#y_train = np.ravel(y_train_df)
#y_test = np.ravel(y_test_df)

# # ================ 2: Keras ================
# Train and Test
# Setup the parameters for NN
in_lay_s  = 3;     # 3 features
hid_lay1_s = 512;   #  number of hidden units layer 1
hid_lay2_s = 128;   #  number of hidden units layer 2
#hid_lay3_s = 32;    #  number of hidden units layer 3
num_lab = 1;            #  1 predicted output
                          
#epochs=500 

# Model Data
# Import `Sequential` from `keras.models`
from keras.models import Sequential

# Import `Dense` from `keras.layers`
from keras.layers import Dense, Dropout

from keras.callbacks import EarlyStopping
#Set early stopping monitor so the model stops training when it won't improve anymore
early_stopping_monitor = EarlyStopping(patience=3)

# Initialize the constructor
model = Sequential()

# Add the input layer 
model.add(Dense(hid_lay1_s, activation='relu', input_shape=(3,)))

# Add one hidden layer 
model.add(Dense(hid_lay2_s, activation='relu'))

# Add one hidden layer 
#model.add(Dense(hid_lay3_s, activation='relu'))

#Add one hidden layer 
#model.add(Dense(32, activation='relu'))

# Add the output layer 
#model.add(Dense(1, activation='sigmoid'))
model.add(Dense(1))

# Model output shape
model.output_shape

# Model summary
model.summary()

# Model config
model.get_config()

# List all weight tensors 
model.get_weights()

# Train model: Compile and Fit
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])             
#model.fit(X_train, y_train, epochs=500, batch_size=1, verbose=1, callbacks=[early_stopping_monitor])
model.fit(X_train, y_train, epochs=1000, batch_size=1, verbose=1)

# Predict 
y_pred = model.predict(X_test)
# Evaluate Model
#score = model.evaluate(X_test, y_test, verbose=1)
# print(score)
#print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))
mse_value, mae_value = model.evaluate(X_test, y_test, verbose=0)

print('MSE:', mse_value)
  
print('RMSE:', math.sqrt(mse_value)) 

print('MAE:', mae_value) 

from sklearn.metrics import r2_score

print('R2 score:', r2_score(y_test, y_pred))

# # ================ 2a: Save the model ================
# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")

# # ================ 3: Visualize and save results ================
fig, ax = plt.subplots()
ax.scatter(y_test, y_pred)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
ax.set_xlabel('Actual Average power (kW)')
ax.set_ylabel('Predicted Average power (kW)')
plt.savefig('comp.pdf')
plt.show()

df_p = df_test.copy()
df_p['Predicted Average power (kW)'] = y_pred

fig, ax = plt.subplots()
#plt.rcParams.update({'figure.figsize':(10,8), 'figure.dpi':100})
ax.scatter(x=df_p['Average wind speed (m/s)'], y=df_p['Average power (kW)'], s= 5, label='Actual')
ax.scatter(x=df_p['Average wind speed (m/s)'], y=df_p['Predicted Average power (kW)'], s= 5, color='red', label='Model Predicted')
ax.set_xlabel('Average wind speed (m/s)')
ax.set_ylabel('Average power (kW)')
ax.legend()
plt.savefig('pred.pdf')
plt.show()

def contourplot(model, windspeed, ax):
   # grid = np.mgrid[-0.4:0.4:0.01,0.7:1.6:0.01]
    alpha = np.random.uniform(-0.5, 0.5, 100000)
    TI = np.random.uniform(0, 50, 100000)
    v = np.ones(100000) * windspeed
    X_test = pd.DataFrame({'Average wind speed (m/s)': v,
                           'Input turbulence intensity (%)': TI,
                           'Input alpha': alpha})
    y_test = model.predict(X_test).flatten()
    cntr = ax.tricontourf(alpha, TI, y_test, levels=50)
    ax.set_xlabel('alpha (-)')
    ax.set_ylabel('TI [%]')
    ax.set_title(f'Predicted power (kW), v={windspeed} (m/s)')
    fig.colorbar(cntr, ax=ax)

fig, [ax0, ax1, ax2] = plt.subplots(1, 3, figsize=(16, 5))
contourplot(model,  7.5, ax0)
contourplot(model, 11.5, ax1)
contourplot(model, 17.5, ax2)
plt.savefig('cont.pdf')
plt.show()