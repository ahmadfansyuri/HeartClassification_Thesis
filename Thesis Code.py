# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 23:26:40 2021

@author: Calon Direktur;
"""

import numpy as np
import pandas as pd
import os
from wfdb import io
import pywt
from statsmodels.robust import mad
from wfdb.processing import normalize_bound

#Import DataFrame records information and drop unused label;
dfRecords = pd.read_csv('D:/Kuliah/RISET/hasil/df_records.csv') 
dfRecords = dfRecords.drop('Unnamed: 0', 1)                     


#Import Physikalisch-Technische Bundesanstalt (PTB) Dataset
def PTBDataBase(df_records):
    
    heartSignalData  = []
    heartSignalLabel = []
    heartSignalClass = ['Bundle branch block']
    
    for i in range(len(heartSignalClass)):
        for j in range(len(df_records)):
            if(df_records.label[j] == heartSignalClass[i]):
                temp = df_records.iloc[j]
                temp = io.rdrecord(record_name=os.path.join('D:/Kuliah/RISET/hasil/ptbdb', temp['name'])).p_signal  
                temp = np.transpose(temp)
                temp = temp[1]
                heartSignalData.append(temp) 
                heartSignalLabel.append(i)
                
    return heartSignalData, heartSignalLabel

#Normalize Signal with y min value = 0 and y max value = 1
def normalizing(signalFeature):
    
    for i in range(len(signalFeature)):
        signalFeature[i] = normalize_bound(signalFeature[i], lb=0, ub=1)
        
    return signalFeature

#Segment signal into 1000/2000/3000/4000 Nodes 
def segmentationSignal(signalFeature, signalLabel):
    
    signalFeatureSegmented  = []
    signalLabelSegmented    = []
    
    for i in range(len(normalizeSignal)):
            print(i)
            
            temp  = normalizeSignal[i]
            label = signalLabel[i]
            
            for j in range(0, len(normalizeSignal[i]), 2000):
                #print(j)
                if (j + 2000 < len(normalizeSignal[i])):
                    signalFeatureSegmented.append(temp[j:j+2000])
                    signalLabelSegmented.append(label)
                
    return signalFeatureSegmented, signalLabelSegmented

#DISCRETE WAVELET TRANSFORM function
def DWT(mf, signalFeatureSegmented):
    result = []
    
    for i in range(len(signalFeatureSegmented)):
        
        data = []
        
        for j in range(len(mf)):
            
            w=pywt.Wavelet(mf[j])
            ca=[]  #Coefficient Aproximation
            cd=[]  #Coefficient Detail
            levels=8 
            temp = signalFeatureSegmented[i]
            
            for level in range(levels):
                (l, h)=pywt.dwt(temp, w) #l is low pass value; h is high pass value;
                ca.append(l)
                cd.append(h)
                
            l = [0]*len(l)
            l=np.array(l)
            cd.append(l)
        
            tho = mad(cd[0])
            uthr = tho*np.sqrt(2*np.log(len(signalFeatureSegmented[i])))
            
            new_cd = []
            for h in cd :
                new_cd.append(pywt.threshold(h, value=uthr, mode='soft')) #soft thresolding
            
            new_cd.reverse()
            new_signal = pywt.waverec(new_cd, w)
            data.append(new_signal)
            x = np.asarray(data)
            x= x.transpose()
            x = x.flatten()
            result.append(x)
            
    return result


#Main codes
signalFeature = []
signalLabel   = []

#Import Data
signalFeature, signalLabel = PTBDataBase(dfRecords)

#Normalize Signal
normalizeSignal = normalizing(signalFeature)

#Segment signal
signalFeatureSegmented  = []
signalLabelSegmented    = []

signalFeatureSegmented, signalLabelSegmented = segmentationSignal(normalizeSignal, signalLabel)

#Calling Wavelet Function
mf = ['sym8'] #Mother Wavelet function

signalDenoising = []
signalDenoising = DWT(mf, signalFeatureSegmented)