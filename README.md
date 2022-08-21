# SMB-Gen
Statistical downscaling of Greenland Ice Sheet Surface Mass Balance.

This project utilizes scikit-learn, netCDF4, matplotlib and stats libraries to clean and transform the multi-dimensional meteorological sattellite data of Greenland.
The AI models used are Deep Neural Networks and Gradient Boosted Decision Trees to predict the changes to Greenland's Ice Sheet. 

Please run the files in the following order
1. createFeature.py
2. createSMB.py
3. dataPreprocessing.py
4. DANN.py/xgBoost.py
