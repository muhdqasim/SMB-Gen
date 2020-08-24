import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.layers import LSTM
import torch
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
from keras.utils.vis_utils import plot_model
import pickle5 as pickle
from keras.callbacks import ModelCheckpoint
from matplotlib import pyplot as plt

pathToTrainingData = '/nfs/annie/sc19mq/dataFiles/trainDataset.pkl'
pathToValidationData = '/nfs/annie/sc19mq/dataFiles/validDataset.pkl'
pathToTestData = '/nfs/annie/sc19mq/dataFiles/testDataset.pkl'
pathToLargeTestData = '/nfs/annie/sc19mq/dataFiles/testDataset_2.pkl'

train = pd.read_pickle(pathToTrainingData)
valid = pd.read_pickle(pathToValidationData)
test = pd.read_pickle(pathToLargeTestData)

#Drop multicollinear RLUT AND TS & non-stationary HUS
featuresRemoved = ["hus", "ts","rlut"]
train.drop(featuresRemoved, axis=1, inplace=True)
valid.drop(featuresRemoved, axis=1, inplace=True)
test.drop(featuresRemoved, axis=1, inplace=True)

print()
print(train)
print()
print(valid)
print()
print(test)
print()


#seperate training features from target features
train_x = train.iloc[:, :7].values
train_y = train.iloc[:,-1:].values

valid_x = valid.iloc[:, :7].values
valid_y = valid.iloc[:,-1:].values

test_x = test.iloc[:, :7].values
test_y = test.iloc[:,-1:].values

dataset_train_x = np.array(train_x, dtype='float32')
dataset_train_y = np.array(train_y, dtype='float32')

dataset_valid_x = np.array(valid_x, dtype='float32')
dataset_valid_y = np.array(valid_y, dtype='float32')

dataset_test_x = np.array(test_x, dtype='float32')
dataset_test_y = np.array(test_y, dtype='float32')


# reshape input to be [samples, time steps, features]
#trainX = np.reshape(dataset_train_x, (dataset_train_x.shape[0], 1, dataset_train_x.shape[1]))
#validX = np.reshape(dataset_valid_x, (dataset_valid_x.shape[0], 1, dataset_valid_x.shape[1]))
#testX = np.reshape(dataset_test_x, (dataset_test_x.shape[0], 1, dataset_test_x.shape[1]))


model = Sequential()
#model.add(LSTM(100, input_shape=(1, trainX.shape[2]) ))
model.add(tf.keras.layers.Dense(2000, activation="tanh"))
model.add(tf.keras.layers.Dense(1000, activation="tanh"))
model.add(tf.keras.layers.Dense(500, activation="tanh"))
model.add(Dense(1))


#Uncomment lines below to load a saved model for testing instead of training and comment out model.fit line below
model.build((1,7))
model.load_weights("/nfs/annie/sc19mq/dataFiles/dann_NEW_BEST.h5")

#opt = tf.keras.optimizers.Adam(learning_rate=0.001)
#model.compile(optimizer=opt, loss= "mse", metrics=['mse'])

#To create model checkpoints in between training
#checkpoint = ModelCheckpoint("/nfs/annie/sc19mq/dataFiles/dann_newSet_ckpt.hdf5", monitor='val_loss', verbose=1,
#                             save_best_only=True, mode='auto', period=1)

# fit network
#history = model.fit(
#  dataset_train_x,
#  dataset_train_y, 
#  epochs=15, 
#  validation_data=(dataset_valid_x, dataset_valid_y), 
#  verbose=2,
#  callbacks=[checkpoint],
#  shuffle=True,
#  use_multiprocessing=True,
#  workers=4
#)
   
#model.save('/nfs/annie/sc19mq/dataFiles/dann_NEW_BEST.h5')

actual_x = np.array([test_x[i] for i in range(0, len(test_x), 10000)]) 
actual_y = np.array([test_y[i] for i in range(0, len(test_y), 10000)])

print("ACTUAL X -> ", actual_x.shape) 
print("ACTUAL Y -> ", actual_y.shape)

y_pred = model.predict(actual_x)

performanceMetric = r2_score(actual_y, y_pred)
print("Performance metric R2 score is: ")
print(performanceMetric)

mseLoss =  mean_squared_error(actual_y, y_pred)
print("Performance metric MSE score is: ")
print(mseLoss)

maeLoss =  mean_absolute_error(actual_y, y_pred)
print("Performance metric MAE score is: ")
print(maeLoss)


#plot_model(model, to_file='/nfs/annie/sc19mq/imageFiles/DANN_model_plot_BEST.png', show_shapes=True, show_layer_names=True)


hist_json_file = '/nfs/annie/sc19mq/dataFiles/DANN_history_BEST.pkl'

#with open(hist_json_file, mode='wb') as f:
#  pickle.dump(history.history, f)

#history = pd.read_pickle(hist_json_file)
  
#print(history.history)
  
# summarize history for loss
#plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
#plt.title('model loss')
#plt.ylabel('loss')
#plt.xlabel('epoch')
#plt.legend(['train', 'valid'], loc='upper right')
#plt.show()


#plt.scatter(test_y, test_y, c='tab:blue', alpha=0.3, label="Actual")
#plt.scatter(y_pred, y_pred, c='tab:red', alpha=0.3, label="Observed")
#plt.title('model loss')
#plt.ylabel('SMB value')
#plt.xlabel('SMB value')
#plt.legend(loc='upper left')
#plt.show()
 

fig, ax = plt.subplots()
ax.plot(actual_y, actual_y, label='Actual Values', c="tab:blue")
ax.scatter(y_pred, y_pred, label='Predicted value', alpha=0.5, c="tab:red", marker="1")
ax.legend()
plt.ylabel('SMB value')
plt.xlabel('SMB value')
plt.title('Actual vs Observed Plot')
plt.show()























