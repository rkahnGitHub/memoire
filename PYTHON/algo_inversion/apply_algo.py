# -*- coding: utf-8 -*-
"""
@author: hassan.bazzi
"""

import numpy as np
import os
import keras
from keras.models import model_from_json
import pandas as pd


PATH = os.getcwd()+'/'
json_file = open(PATH+'modeldry.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights(PATH+"modeldry.h5")
print("Loaded model from disk")

###Open the text file to load the data
zonal_stat = "stat_25052021.csv"
dataset_val=pd.read_csv(PATH+zonal_stat,delimiter=',')
id_ =dataset_val.iloc[:,0].values
inc_val = dataset_val.iloc[:,2].values
VV_val = 10*np.log10(dataset_val.iloc[:,1].values)
veg_val = (dataset_val.iloc[:,3].values)/100
inputs_val = np.column_stack((inc_val,veg_val,VV_val))

###Apply the algorithm
estimation= loaded_model.predict(inputs_val).flatten()
###Save the results in a text file
file_out = open(PATH+"resultats_TPcopernicus"+zonal_stat[:-4]+".txt",'w')
file_out.write('ID_PARCEL'+'\t'+'Estimation'+'\n')
for e in range(len(estimation)):
    file_out.write(str('%.0f' %id_[e])+'\t'+str('%.2f' %estimation [e])+"\n")

file_out.close()
