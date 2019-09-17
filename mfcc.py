################################################
#Linda Adams - Freesound Audio Tagging Challenge
################################################

#Mel Frequency Cepstral Coefficients

################################################

import numpy as np 
import pandas as pd 

import os
print(os.listdir("Data"))
import librosa

sr = 44100

#training wavs
audio_train_files = os.listdir('Data/audio_train')
#test wavs
audio_test_files = os.listdir('Data/audio_test')

#dataframe of training data list
train = pd.read_csv('Data/train.csv')
#dataframe of testing data list
submission = pd.read_csv('Data/sample_submission.csv')


#function for getting mfcc
def get_mfcc(name, path):
    b, _ = librosa.core.load(path + name, sr = sr)
    assert _ == sr
    try:
        gmm = librosa.feature.mfcc(b, sr = sr, n_mfcc=20)
        return pd.Series(np.hstack((np.mean(gmm, axis=1), np.std(gmm, axis=1))))
    except:
        print('bad file')
        return pd.Series([0]*40)
    
    
train_data = pd.DataFrame()
train_data = train['fname'].apply(get_mfcc, path="Data/audio_train/")
train_data['label'] = train['label']  
train_data.to_csv("TRAINCSV.csv")

test_data = pd.DataFrame()
test_data = submission['fname'].apply(get_mfcc, path="Data/audio_test/")
test_data['label'] = submission['label']  
test_data.to_csv("TESTCSV.csv")




################################################################
train_data = pd.read_csv("train_data.csv", header = None)

test_data = test_data['fname'].apply(get_mfcc, path="Data/audio_test/")

test_data['label'] = train['label']  

test_data = pd.read_csv("test_data.csv", header = None)


######################################################################

b, _ = librosa.core.load('Data/audio_train/00044347.wav', sr = sr)

gmm_1 = librosa.feature.mfcc(b, sr = sr, n_mfcc=20)  

first_row = pd.Series(np.hstack((np.mean(gmm_1, axis=1), np.std(gmm_1, axis=1))))

list_1 = gmm_1[0]

#train_data.to_csv("practice_test.csv")

#############################################

X = train_data.drop('label', axis=1)
feature_names = list(X.columns)
X = X.values
labels = np.sort(np.unique(train_data.label.values))
num_class = len(labels)
c2i = {}
i2c = {}
for i, c in enumerate(labels):
    c2i[c] = i
    i2c[i] = c
y = np.array([c2i[x] for x in train_data.label.values])



    
    
    