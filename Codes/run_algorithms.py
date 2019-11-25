#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 16:39:33 2019

@author: sadat
"""
import os
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.metrics import recall_score
from sklearn.model_selection import LeaveOneGroupOut
from keras.utils import np_utils
import keras    
from sklearn.metrics import confusion_matrix
from keras.callbacks import EarlyStopping
from keras.layers import Bidirectional
from keras.layers import Masking
from keras.layers import Dropout

class run_sequential_learning:
    """
    This class will run the deep learning algorithms, LSTM and BLSTM
    from the speakerwise processed data
    """
    def __self__(self, algorithm):
        self.algorithm = algorithm
        
    def prepare_data(self, directory):
        #Function prepares data from the saved DataFrame
        Features = {} #This dictionary will stack the features across different speaker
        Labels = [] # List for stacking forecasting labels
        Speakers = [] #List for tracking speaker id
        for speaker in range(1,11):
            filepath = os.path.join(directory,'audio_visual_speaker_{}.csv'.format(speaker))
            feature_set = pd.read_pickle(filepath)
            Features[speaker]=np.vstack(([feature_set['features'][idx] for idx in range(feature_set.shape[0])]))
            Labels = Labels + [feature_set['UF_label'][idx] for idx in range(feature_set.shape[0])]
            Speakers = Speakers + [speaker for idx in range(feature_set.shape[0])]
            
        Full_feature_set = np.vstack(([Features[speaker] for speaker in range(1,11)])) #Putting all the features together
        
        #We will reshape the features in three-dimensions. The first dimension 
        #is number of data instances, which is equal to total data point. Second
        #is, the length of sequence, which is equal fdr all utterances and feature
        #dimension is the third dimension which is equal to 895
        data_instances = len(Labels)
        sequence_length = int(Full_feature_set.shape[0]/data_instances)
        feature_dimension = 895
        Full_feature_set = Full_feature_set.reshape(data_instances, 
                                                    sequence_length, 
                                                    feature_dimension)
        return Full_feature_set, np.asarray(Labels), np.asarray(Speakers)
    
    
    def LSTM(self, features, labels, speaker_id, model_type):
        """
        This block will make LSTM cells and produce output
        """
        logo = LeaveOneGroupOut()
        #code for LSTM model
        LSTM_predict = {} #saves the softmax output
        LSTM_test = {} #saves the test set GTs
        LSTM__modelpred = {} #saves the final outputs
        LSTM_con = {} #saves the confusion matrix per speaker
        LSTM_UWR = {} #saves the unweighted recall per speaker
        speaker = 0
        for train, test in logo.split(features, labels, speaker_id):
            label_train = np_utils.to_categorical(labels[train])
            # Set callback functions to early stop training 
            callbacks= [EarlyStopping(monitor='val_loss', patience=10)]    
            model = Sequential()
            model.add(Masking(mask_value=0., input_shape=(features.shape[1], features.shape[2])))
            if model_type=='BLSTM':
                model.add(Bidirectional(LSTM(128, return_sequences=True)))
                model.add(Dropout(0.5))
                model.add(Bidirectional(LSTM(128)))
            elif model_type=='LSTM':
                model.add(LSTM(128, return_sequences=True))
                model.add(Dropout(0.5))
                model.add(LSTM(128))
            else:
                print('error in model type')
                break
            model.add(Dropout(0.5))
            model.add(Dense(256, activation='relu'))
            model.add(Dropout(0.5))##
            model.add(Dense(4, activation='softmax'))
            adam=keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)   
            model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
            model.fit(features[train], label_train,  epochs=50, validation_split=0.20, callbacks=callbacks, batch_size=128, verbose=1)
            X_pred = model.predict(features[test,:])
            LSTM_predict[speaker] = X_pred
            LSTM_test[speaker] = labels[test]
            Y_pred = np.argmax(X_pred, axis=1)
            LSTM_con[speaker] = confusion_matrix(labels[test],Y_pred)
            LSTM_UWR[speaker] = recall_score(labels[test],Y_pred, average='macro') 
            LSTM_predict[speaker] = X_pred
            LSTM_test[speaker] = labels[test]
            LSTM__modelpred[speaker] = Y_pred
            LSTM__modelpred[speaker] = Y_pred
            speaker+=1
        
            
        
            
        
        
        
        