import pandas as pd
import xarray as xs
import numpy as np
import glob

pathToSMBFiles = glob.glob("/nfs/annie/shared/SMB-Gen/ML_downscaling/mar_data/CSIRO-Mk3.6-*/outputs/MARv3.9-yearly-CSIRO-Mk3.6-*")

#Combine SMB files by coordinates
datasetSMB = xs.open_mfdataset(pathToSMBFiles, combine='by_coords', decode_times= False)
  
print()
print()
print(datasetSMB)
print()
print()
print(datasetSMB.variables)
print()
print()

#Resample to year starting from 1950
datasetSMB['time'] = pd.date_range(start= "1950-12-31", periods=datasetSMB.sizes['time'], freq='Y')

#Drop unnecessary columns
datasetSMB = datasetSMB.drop(['MSK', 'SRF','AF', 'dSMB', 'RU','dRU','ST', 'dST'])

print()
print()
print(datasetSMB)
print()
print()
print(datasetSMB.variables)
print()
print()

#Convert to pandas dataframe
datasetSMB_DF = datasetSMB.to_dataframe()
#Remove NaN and negative fill values
datasetSMB_DF = datasetSMB_DF[datasetSMB_DF.replace([np.inf, -np.inf], np.nan).notnull().all(axis=1)]

#Rearrange index
datasetSMB_DF = datasetSMB_DF.reset_index(['time','x','y'], drop = False)
datasetSMB_DF.drop(["x","y"], inplace=True, axis=1)
datasetSMB_DF = datasetSMB_DF.set_index(['time','LON','LAT'])

#Save dataframe for further processing
datasetSMB_DF.to_pickle("/nfs/annie/sc19mq/dataFiles/SMB_ALLYEARS_DF_NEW.pkl")
