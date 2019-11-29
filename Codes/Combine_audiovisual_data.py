#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 14:41:12 2019

@author: sadat
"""
import scipy.io as sio
import pandas as pd
from scipy import interpolate
import numpy as np
import pickle
import re
import os


class combining_AV:
    """
    This class works for cobining the audio and visual data of IEMOCAP dataset
    """
    def __init__(self, audio_feature_loacation, video_feature_location):
        self.audio_feature_loacation = audio_feature_loacation
        self.video_feature_location = video_feature_location
        print('Combining the feature set...')
    
    def load_files(self, session, name):
        
        #Pitch
        pitch = os.path.join(self.audio_feature_loacation, 's{}'.format(session), 
                     '{}.pitch'.format(name))
        P = pd.read_csv(pitch, header = None)[0].to_list()
        P = [float(value) for count, value in enumerate(P) if count>=4]
        # Energy or intensity
        energy = os.path.join(self.audio_feature_loacation, 's{}'.format(session), 
                      '{}.intensity'.format(name))
        E = pd.read_csv(energy, header = None)[0].to_list()
        E = [float(value) for count, value in enumerate(E) if count>=4]
        #MFb
        mfb = os.path.join(self.audio_feature_loacation, 's{}'.format(session), 
                   '{}.mfb'.format(name))
        MFB = pd.read_csv(mfb, header = None)[0].to_list()
        List_MFB_string = [re.findall("[+-]?\d+\.\d+", i) for i in MFB]
        MFB_value=[]
        for l in List_MFB_string:
            MFB_value.append([float(i) for i in l])
            
        #MFCC
        mfcc = os.path.join(self.audio_feature_loacation, 's{}'.format(session), 
                    '{}.mfcc'.format(name))
        MFCC = pd.read_csv(mfcc, header = None)[0].to_list()
        List_MFCC_string = [re.findall("[+-]?\d+\.\d+", i) for i in MFCC]
        MFCC_value=[]
        for l in List_MFCC_string:
            MFCC_value.append([float(i) for i in l])
    
        return P, E, MFB_value[4:], MFCC_value[4:]
    

    def downsample_data(self, data, value):
        del_row = [i for i in range(len(data)) if i%value!=0]
        new_data = np.delete(data, del_row, axis=0)
        return new_data   
    
    
    def fill_missing_value(self, matrix):
        """
        This function will fill the missing nan values in a matrix.
        It will be used when we find less than 30% missining value
        """
        x = np.arange(0, matrix.shape[1])
        y = np.arange(0, matrix.shape[0])
        #mask invalid values
        matrix = np.ma.masked_invalid(matrix)
        xx, yy = np.meshgrid(x, y)
        #get only the valid values
        x1 = xx[~matrix.mask]
        y1 = yy[~matrix.mask]
        newarr = matrix[~matrix.mask]
        
        filled_matrix = interpolate.griddata((x1, y1), newarr.ravel(),
                                  (xx, yy),
                                     method='cubic')
        return filled_matrix
        
        
    def produce_speakerwise_AV_data(self):
        """
        This code will produce speaker-wise audio-visual data with a same framerate.
        Also, it will operate on the NaN features and remove or replace them.
        """
        
        
        for speaker in range(1,11):
            Audio_Visual = pd.DataFrame(columns=['name','video','audio','label'])
            filename = os.path.join(self.video_feature_location, "All_Audio_visual_s_{}.mat".
                            format(speaker))
            v_data = sio.loadmat(filename)['All_data']
            
            # speaker number even is male and odd is female. 
            #season 1 contains speaker 1 and 2 and 
            #season 2 contains speaker 3 and 4 and so on.. 
            if speaker%2==0:
                session = int(speaker/2)
                gender = 'M'
            else:
                session = int((speaker + 1)/2)
                gender = 'F'
                
            #Next, we will loop through all the data in v_data
            for data in range(v_data.shape[1]): #for data in range(v_data.shape[1]):
                if v_data[0][data]['name'][0][5]==gender: 
                    #This condition is to make sure it's not a 'dataless' utterance
                    filename = v_data[0][data]['name'][0]
                    Pitch, Energy, MFB, MFCC = self.load_files(session, filename)
                    #Next, we see how many nan values in each utterance and process them
                    NaN_loc = np.argwhere(np.isnan(v_data[0][data]['video']))
                    item, count = np.unique(NaN_loc[:,1], return_counts=True)
                    try:
                        percent_nans = np.max(count)/len(v_data[0][data]['video'])
                    except:
                        percent_nans = 0
                        
                    if any([P>0 for P in Pitch]) and percent_nans<0.3:
                        #Our main focus is pitch. For the start and end zeros of pitch
                        #means, there are no information. 
                        
                        Non_zero_pitch = [loc for loc, val in enumerate(Pitch) if val!=0]
                        start = min(Non_zero_pitch)
                        
                        finish = min([max(Non_zero_pitch) + 1, len(Energy), len(MFB), len(MFCC)])
                        Length = finish - start 
                        downsampled_video = self.downsample_data(v_data[0][data]['video'], 3) 
                        #The videodata is extrcated in a framerate three times
                        #of the audio data
                        nan_removed_video = self.fill_missing_value(downsampled_video)

                        audio_data_matrix = np.concatenate((np.asarray(Pitch[start:finish]).reshape(1, Length),
                                                      np.asarray(Energy[start:finish]).reshape(1, Length), 
                                  np.transpose(np.asarray(MFB[start:finish])), np.transpose(np.asarray(MFCC[start:finish]))), axis = 0)
                        audio_data_matrix = np.transpose(audio_data_matrix)                 
                        av_data = {'name':v_data[0][data]['name'][0], 'video':nan_removed_video[start:finish],
                                   'audio':audio_data_matrix, 'label':v_data[0][data]['categorical'][0][0]}
                        Audio_Visual = Audio_Visual.append(av_data, ignore_index=True)
                        print('processed upto speaker {} and data {}'.format(speaker, data))
                        
            if not os.path.exists('Files/sameframe'):
                os.mkdir('Files/sameframe')
                
            Audio_Visual.to_pickle('Files/sameframe/audio_visual_speaker_{}.csv'.format(speaker))   
            
                        
                        
            
#### Main function code testing ####
#Data = combining_AV('/home/sadat/kimlab/Sadat/IEMOCAP_forcasting/audio_features', 
#                    '/home/sadat/kimlab/Sadat/IEMOCAP_forcasting/Old_Analysis/Audio_video_all/AV_all_ORIGINAL')
#Data.produce_speakerwise_AV_data()



