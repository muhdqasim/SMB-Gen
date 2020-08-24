import pandas as pd
import xarray as xs
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor  
from scipy import stats
from statsmodels.tsa.stattools import adfuller
from scipy.spatial.distance import cdist
from sklearn.neighbors import BallTree
import itertools
import collections
import folium
from tqdm import tqdm
from statsmodels.tools.tools import add_constant
from matplotlib import pyplot
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
    
    
pathToFeatures_DF = "/nfs/annie/sc19mq/dataFiles/FEATURE_ALLYEARS_DF_NEW.pkl"
pathToSMB_DF = "/nfs/annie/sc19mq/dataFiles/SMB_ALLYEARS_DF_NEW.pkl"

smb = pd.read_pickle(pathToSMB_DF)
feature = pd.read_pickle(pathToFeatures_DF)

print()
print()
print("--------------PRINT rows in STANDARD FEATURE DATASET-----------------------------")
print()
print(feature)
print()
print(feature.shape)
print()
print()

print()
print()
print("--------------PRINT rows in STANDARD SMB DATASET-----------------------------")
print()
print(smb)
print()
print(smb.shape)
print()
print()

#Check and clean the outliers in feature dataset
spread = feature.describe().T
IQR = spread['75%'] - spread['25%']
spread['outliers'] = (spread['min']<(spread['25%']-(3*IQR)))|(spread['max'] > (spread['75%']+3*IQR))  
z_scores = stats.zscore(feature)
abs_z_scores = np.abs(z_scores)
filtered_entries = (abs_z_scores < 3).all(axis=1)
clean_features = feature[filtered_entries]

print()
print()
print("--------------PRINT rows in OUTLIER CLEANED DATASET-----------------------------")
print()
print(clean_features)
print()
print(clean_features.shape)
print()
print()

#Check and clean the outliers in SMB dataset
spread = smb.describe().T
IQR = spread['75%'] - spread['25%']
spread['outliers'] = (spread['min']<(spread['25%']-(3*IQR)))|(spread['max'] > (spread['75%']+3*IQR))  
z_scores = stats.zscore(smb)
abs_z_scores = np.abs(z_scores)
filtered_entries = (abs_z_scores < 3).all(axis=1)
clean_smb = smb[filtered_entries]

print()
print()
print("--------------PRINT rows in OUTLIER CLEANED SMB-----------------------------")
print()
print(clean_smb)
print()
print(clean_smb.shape)
print()
print()



#Check for correlation in the cleaned feature file
clean_features = add_constant(clean_features)
vif = pd.Series([variance_inflation_factor(clean_features.values, i) 
               for i in tqdm(range(clean_features.shape[1]))], 
              index=clean_features.columns)
   
           
print("-----------------VIF MULTCOLLINEARITY VALUE--------------------")
print(vif)
print()

#Features RLUT and TS have high multiconllinearity, they will be removed later

#Check For Histogram and mean plots to detect stationarity
clean_features.reset_index(inplace=True)
clean_features.set_index('time', inplace=True)

#Group by mean value
grouped = clean_features.groupby(['time']).mean()

print()
print(grouped)
print()

features = ["ps", "cl", "hus", "pr", "psl", "rsut", "ta"]

for feature in features:
  grouped[feature].plot(title='Mean value over the years', colormap='jet')
  pyplot.show()
  
  grouped[feature].hist()
  pyplot.show()


#DICKEY FULLER TEST FOR STATIONARITY, MORE ROBUST THAN CHECKING HISTOGRAM AND PLOTS
for i in tqdm(range(len(features))):
  print("------DICKEY FULLER TEST FOR ", features[i], " ------------------")
  dftest = adfuller(clean_features[features[i]], autolag='AIC')
  dfoutput = pd.Series(dftest[:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
  for key, value in dftest[4].items():
    dfoutput['Critical Value (%s)'%key] = value
  del dftest
  del dfoutput
  

#eature HUS is found to be non-stationary. It will be removed later along with highly collinear RLUT and TS

#Uncomment line below to save progress till now
#clean_features.to_pickle("/nfs/annie/sc19mq/dataFiles/featuresGlobal_UNCORR_OUTLIERS.pkl")
#clean_smb.to_pickle("/nfs/annie/sc19mq/dataFiles/smbRegional_OUTLIERS.pkl")

#Uncomment to resume from previous checkpoint
#clean_features = pd.read_pickle("/nfs/annie/sc19mq/dataFiles/featuresGlobal_UNCORR_OUTLIERS.pkl")
#clean_smb = pd.read_pickle("/nfs/annie/sc19mq/dataFiles/smbRegional_OUTLIERS.pkl")

print()
print()
print("--------------PRINT rows in OUTLEIR REMOVED FEATURE DATASET AFTER CLEANSING-----------------------------")
print()
print(clean_features)
print()
print(clean_features.shape)
print()
print()
print()
print()

print()
print()
print("--------------PRINT rows in OUTLEIR REMOVED SMB DATASET AFTER CLEANSING-----------------------------")
print()
print(clean_smb)
print()
print(clean_smb.shape)
print()
print()
print()
print()


#Get cluster score using elbow  method
K_clusters = range(1,10)
kmeans = [KMeans(n_clusters=i) for i in K_clusters]
Y_axis = np.array(clean_smb['LAT']).reshape(-1,1)
X_axis = np.array(clean_smb['LON']).reshape(-1,1)
score = [kmeans[i].fit(Y_axis).score(Y_axis) for i in range(len(kmeans))]

for i in range(len(kmeans)):
  score = kmeans[i].fit(Y_axis).score(Y_axis)
  print("CLUSTERS: ", i)
  print("SCORE: ", score)
  print()
  



#Cluster using 5 clusters
kmeans = KMeans(n_clusters = 5, init ='k-means++')
clean_smb['cluster_label'] = kmeans.fit_predict(clean_smb[clean_smb.columns[:2]])
clean_features['cluster_label'] = kmeans.predict(clean_features[clean_features.columns[:2]])


#Uncomment this line to save progress till now
#clean_smb.to_pickle("/nfs/annie/sc19mq/dataFiles/SMBRegional_ALLYEARS_CLUSTERED_5.pkl")
#clean_features.to_pickle("/nfs/annie/sc19mq/dataFiles/featuresGlobal_ALLYEARS_CLUSTERED_5.pkl")



#uncomment this line to read from previously stores files
#clean_smb = pd.read_pickle("/nfs/annie/sc19mq/dataFiles/SMBRegional_ALLYEARS_CLUSTERED_5.pkl")
#clean_features = pd.read_pickle("/nfs/annie/sc19mq/dataFiles/featuresGlobal_ALLYEARS_CLUSTERED_5.pkl")

clean_features = clean_features.groupby(['time','cluster_label'])["ps", "cl", "hus", "pr", "psl", "rsut", "ta"].mean()

clean_smb = clean_smb.groupby(['time','cluster_label'])['SMB'].mean()

datasetMerged = clean_features.join(clean_smb, how='outer')

datasetMerged = datasetMerged.dropna()

#Normalize dataset for faster machine learning training
features_std = StandardScaler().fit_transform(datasetMerged)
datasetMerged[:] =  features_std
datasetMerged = datasetMerged.round(decimals=3)

#datasetMerged.to_pickle('/nfs/annie/sc19mq/dataFiles/finalFile.pkl')

#dataset_train = pd.read_pickle('/nfs/annie/sc19mq/dataFiles/finalFile.pkl')

dataset_train = datasetMerged.sample(frac=1).reset_index(drop=True)


print("AFTER SHUFFLING")
print(dataset_train)
print()

lenTrain = round(len(dataset_train.index) * 0.9)
actualTrain = round(lenTrain * 0.9)

print("LEN TRAIN    -> ", lenTrain)
print("ACTUAL TRAIN    -> ", actualTrain)

train_dataset = dataset_train.iloc[:actualTrain]
valid_dataset = dataset_train.iloc[actualTrain:lenTrain]
test_dataset = dataset_train.iloc[lenTrain:]

print("TRAIN")
print(train_dataset)

print("VALID")
print(valid_dataset)

print("TEST")
print(test_dataset)

train_dataset.to_pickle("/nfs/annie/sc19mq/dataFiles/trainDataset.pkl")
valid_dataset.to_pickle("/nfs/annie/sc19mq/dataFiles/validDataset.pkl")
test_dataset.to_pickle("/nfs/annie/sc19mq/dataFiles/testDataset.pkl")

