# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 19:26:40 2021

@author: jchet
"""

import pandas as pd
import numpy as np

path_dataset = "./src/data.csv"

path_address10K = "./parsed_data/address_10K.txt"
path_address100K = "./parsed_data/address_100K.txt"
path_address1M = "./parsed_data/address_1M.txt"

path_name10K = "./parsed_data/name_10K.txt"
path_name100K = "./parsed_data/name_100K.txt"
path_name1M = "./parsed_data/name_1M.txt"


def exportAddress (readFilePath, writeFilePath, numberOfAddress, person_address):
    
    dataset = pd.read_csv(readFilePath, sep=";",na_filter= False)
    mydataset = dataset.replace(np.nan, '', regex=True)
    
    if person_address == True:
         address = mydataset['person_address']
         
    else:
        address = mydataset['person_name']
        
    array = address.values
    
    if numberOfAddress == "10K":
        newFile = open(writeFilePath,"wb")
        array = address.values
        array_10k = []
        count = 0
        for row in array:
            if count<10000:
                array_10k.append(row)
                count += 1
        np.savetxt(newFile, array_10k, fmt='%s', delimiter=';', newline='\n', header='', footer='', comments='# ', encoding="utf-8")
        newFile.close()
        
    elif numberOfAddress == "100K":
        newFile = open(writeFilePath,"wb")
        array = address.values
        array_100k = []
        count = 0
        for row in array:
            if count<100000:
                array_100k.append(row)
                count += 1
        np.savetxt(newFile, array_100k, fmt='%s', delimiter=';', newline='\n', header='', footer='', comments='# ', encoding="utf-8")
        newFile.close()
        
    else:
        newFile = open(writeFilePath,"wb")
        array = address.values
        np.savetxt(newFile, array, fmt='%s', delimiter=';', newline='\n', header='', footer='', comments='# ', encoding="utf-8")
        newFile.close()
        
    
if __name__ == '__main__':        
    exportAddress(path_dataset,path_address10K,"10K",True)
    exportAddress(path_dataset,path_address100K,"100K",True)
    exportAddress(path_dataset,path_address1M,"",True)

    exportAddress(path_dataset,path_name10K,"10K",False)
    exportAddress(path_dataset,path_name100K,"100K",False)
    exportAddress(path_dataset,path_name1M,"",False)
