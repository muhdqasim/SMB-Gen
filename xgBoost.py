import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV
from matplotlib import pyplot

pathToTrainingData = '/nfs/annie/sc19mq/dataFiles/trainDataset.pkl'
pathToValidationData = '/nfs/annie/sc19mq/dataFiles/validDataset.pkl'
pathToTestData = '/nfs/annie/sc19mq/dataFiles/testDataset.pkl'

#pathToLargeTestData = '/nfs/annie/sc19mq/dataFiles/MergedDataset_NotStandard.pkl'

train = pd.read_pickle(pathToTrainingData)
valid = pd.read_pickle(pathToValidationData)
test = pd.read_pickle(pathToTestData)

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

#seperate training features from target features
train_x = train.iloc[:, :7].values
train_y = train.iloc[:,-1:].values

valid_x = valid.iloc[:, :7].values
valid_y = valid.iloc[:,-1:].values

test_x = test.iloc[:, :7].values
test_y = test.iloc[:,-1:].values


print("TRAIN SHAPE    ---> ", train_x.shape)
print()
print("VALID SHAPE    ---> ", valid_x.shape)
print()
print("TEST SHAPE    ---> ", test_x.shape)


model = xgb.XGBRegressor(
  max_depth = 8,  
  n_estimators =30,
  learning_rate = 0.2,
  objective = 'reg:squarederror',
  random_state=320,
  booster='dart',
  tree_method ='exact',
  rate_drop = 0.1,
  reg_alpha = 1.3,
  gamma = 1.0,
  colsample_bytree= 1.0,
  n_jobs = 5
)

model.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)], verbose=True, early_stopping_rounds=3)
model.save_model("/nfs/annie/sc19mq/dataFiles/xgModel_1.model")

#model.load_model('/nfs/annie/sc19mq/dataFiles/xgModel_BEST.model')

y_pred = model.predict(test_x)

performanceMetric = r2_score(test_y, y_pred)

print("Performance metric R2 score is: ")
print(performanceMetric)

mseLoss =  mean_squared_error(test_y, y_pred)
print("Performance metric MSE score is: ")
print(mseLoss)

maeLoss =  mean_absolute_error(test_y, y_pred)
print("Performance metric MAE score is: ")
print(maeLoss)

results = model.evals_result()

x_axis = range(0, 17)
# plot error loss
fig, ax = pyplot.subplots()
ax.plot(x_axis, results['validation_0']['rmse'], label='Train')
ax.plot(x_axis, results['validation_1']['rmse'], label='Validation')
ax.legend()
pyplot.ylabel('Root Mean Squared Error')
pyplot.xlabel('Epoch')
pyplot.title('XGBoost Error Loss')
pyplot.show()
