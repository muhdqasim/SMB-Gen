import pandas as pd
import xarray as xs
import glob
import numpy as np

# Get a list of all .nc files available in different folders
filenames = glob.glob("/nfs/annie/shared/SMB-Gen/ML_downscaling/mar_data/CSIRO-Mk3.6-*/input_years/*.nc")

datasetFeature = xs.open_mfdataset(filenames, combine='by_coords')

print()
print()
print(datasetFeature)
print()
print()
print(datasetFeature.variables)
print()
print()


#slice dataset between 1950 to 2100
datasetFeature = datasetFeature.sel(time=slice('1950-01-01', '2100-12-31'))

print("=============AFTER SLICING THE DATASET -----")
print()
print(datasetFeature)
print()
print()
print(datasetFeature.variables)
print()
print()


#drop unnecessary columns
datasetFeature = datasetFeature.drop(['time_bnds', 'lon_bnds','lat_bnds', 'p0', 'lev_bnds','a','b', 'a_bnds', 'b_bnds'])


datetimeindex = datasetFeature.indexes['time'].to_datetimeindex()

datasetFeature['time'] = datetimeindex

#Resample to year and skip NaN values
datasetFeature = datasetFeature.resample(time='Y', restore_coord_dims=True, skipna = True).mean(skipna= True)


print("=============RESAMPLED TO YEAR -----")
print()
print(datasetFeature)
print()
print()
print(datasetFeature.variables)
print()
print()

#Convert to Pandas Dataframe
datasetFeature_DF = datasetFeature.to_dataframe()
#Remove NaN and negative fill values
datasetFeature_DF = datasetFeature_DF[datasetFeature_DF.replace([np.inf, -np.inf], np.nan).notnull().all(axis=1)]

#Rearrange index values
datasetFeature_DF = datasetFeature_DF.reset_index(["lat","lon","lev","plev","time"])
datasetFeature_DF = datasetFeature_DF.set_index(["lat","lon","time","lev","plev"])

print("=============DATAFRAME YEAR -----")
print()
print(datasetFeature_DF)
print()
print()
print()
print()

#Save dataframe file for further processing
pathToFeatures_DF = "/nfs/annie/sc19mq/dataFiles/FEATURE_ALLYEARS_DF_NEW.pkl"
datasetFeature_DF.to_pickle(pathToFeatures_DF)

