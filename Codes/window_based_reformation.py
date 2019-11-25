#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 14:28:43 2019

@author: sadat
"""

import pandas as pd
import os
import numpy as np

class window_based_reformation:
    
    """
    This code converts the frame level data into a window-based reformed data.
    It also prouces the statistical features for those windows
    """
    
    def __init__(self, file_location):
        self.file_location = file_location
        
    def make_window_idx(self, framelength, frame_idx, overlap, window_type):
        """
        This function will crete the index list for making overlapped windows,
        through which, we will produce statistical features.
        for example, make_window_idx(80, 30, 15, window_type='dynamic') returns;
        [[0, 29], [15, 44], [30, 59], [45, 74], [60, 79]]
        If window_type='static', it will return
        [[0, 79]]
        """
        index_list = []
        if window_type=='dynamic':
            i = 0
            j = 0
            while j<framelength-1:
                if (i + frame_idx -1)<framelength:
                    index_list.append([i, i+frame_idx-1])
                    j = i+frame_idx-1
                else:
                    index_list.append([i, framelength-1])
                    break
                i+=overlap
        elif window_type=='static':
            index_list = [[0, framelength-1]]
        return index_list
        
    
    
    def process_data(self, window_type):
        """
        Function which creates the statistical window-based data
        """
        
        for speaker in range(1,11):
            Audio_Visual = pd.DataFrame(columns=['name','stat_features','label'])
            directory = os.path.join(self.file_location, 'audio_visual_speaker_{}.csv'.format(speaker))
            audio_visual_framewise = pd.read_pickle(directory)
            
            for utterance in range(len(audio_visual_framewise)):
                index_list = self.make_window_idx(audio_visual_framewise['audio'][utterance].shape[0], 30, 15, window_type)
                features_concatenated = np.concatenate((audio_visual_framewise['audio'][utterance], 
                                                        audio_visual_framewise['video'][utterance]), axis = 1)
            # Extract statistical information from window-based data
                window_wise_feature = np.zeros((len(index_list), 895)) # Although 179 features, 5 statistical feature from each of them. 
                for idx in range(len(index_list)):
                    parsed_features = features_concatenated[index_list[idx][0]:index_list[idx][1]]
                    statistical_feat = np.concatenate((np.mean(parsed_features, axis=0).reshape(1, 179), 
                                                       np.std(parsed_features, axis=0).reshape(1, 179),
                                                       np.quantile(parsed_features, 0.25, axis=0).reshape(1, 179), 
                                                       np.quantile(parsed_features, 0.75, axis=0).reshape(1, 179), 
                                                       np.quantile(parsed_features, 0.75, axis=0).reshape(1, 179) - 
                                                       np.quantile(parsed_features, 0.25, axis=0).reshape(1, 179)), axis=1)
                    window_wise_feature[idx, :] = statistical_feat
                    print('speaker is {} and data is {}, len of the idx is{}'.format(speaker, idx, len(index_list)))
                feature_set = {'name': audio_visual_framewise['name'][utterance], 
                               'stat_features':window_wise_feature, 
                               'label':audio_visual_framewise['label'][utterance]}
                Audio_Visual = Audio_Visual.append(feature_set, ignore_index=True)
                
            if not os.path.exists('Files/statistical'):
                os.mkdir('Files/statistical')          
            Audio_Visual.to_pickle('Files/statistical/audio_visual_speaker_{}.csv'.format(speaker)) 
                    
                    
                    
                    
#Main function fpr test only

#task = window_based_reformation('Files/sameframe', window_type='dynamic')
#task.process_data()               
            
            
            
            
            