import pandas as pd
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import recall_score

directory = Path('results/bilstm')

time_distances = []
results = {}
for item in os.listdir(directory):
    speaker_id = item.split('_')[-1]
    df = pd.read_pickle(directory / item)
    results[speaker_id] = {col: df[col].iloc[0] for col in df.columns}

    time_distances = time_distances + pd.read_pickle(
        f'Files/UF_His_data/step_1_mean/dynamic_next/audio_visual_speaker_{speaker_id}.csv')['time_distance'].tolist()

print(results.keys())
# print('Forecasts shape:', results['1F']['forecasts'].shape)
# print('Forecasts shape:', results['1F']['true'].shape)
print('Probabilities shape:', results['1F']['probabilities'].shape)
print('Number of labels:', len(results['1F']['true_labels']))

fig, ax = plt.subplots(figsize=(15,15))
ax.scatter(list(range(len(time_distances))), time_distances)
ax.set_ylabel('time_distances', fontsize=30)
ax.set_xlabel('utterance_ids', fontsize=30)
ax.tick_params(axis='both', which='major', labelsize=20)
ax.set_title('Time distance plots for all utterances in step-1 forecasting', fontsize=25, color='r')
# plt.show()

TG_index = {
    1: [index for index, time in enumerate(time_distances) if 0 < time < 6],
    2: [index for index, time in enumerate(time_distances) if 6 < time < 12],
    3: [index for index, time in enumerate(time_distances) if 12 < time < 18]
}

all_gt = []
all_pred = []
for speaker in results.keys():
    all_gt.append(results[speaker]['true_labels'])
    all_pred.append(np.argmax(results[speaker]['probabilities'], axis=1))

all_gt = np.concatenate(all_gt)
all_pred = np.concatenate(all_pred)

print('GT shape:', all_gt.shape)
print('Predictions shape:', all_pred.shape)
print('Time distances length:', len(time_distances))

GT_timewise= {}
Pred_timewise = {}
for time_group in TG_index.keys():
    GT_timewise[time_group] = [all_gt[i] for i in TG_index[time_group]]
    Pred_timewise[time_group] = [all_pred[i] for i in TG_index[time_group]]
    RECALL =  recall_score(GT_timewise[time_group],  Pred_timewise[time_group], average='macro')
    print('Performance for group {0} is {1:.2f}%'.format(time_group, RECALL*100))