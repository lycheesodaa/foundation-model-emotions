import pandas as pd
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt

directory = Path('results/moirai_pl24')

time_distances = []
results = {}
for item in os.listdir(directory):
    speaker_id = item.split('_')[-1]
    df = pd.read_pickle(directory / item)
    results[speaker_id] = {col: df[col].iloc[0] for col in df.columns}

    time_distances = time_distances + pd.read_pickle(
        f'Files/UF_His_data/step_1_mean/dynamic_next/audio_visual_speaker_{speaker_id}.csv')['time_distance'].tolist()

print(results.keys())
print('Forecasts shape:', results['1F']['forecasts'].shape)
print('Probabilities shape:', results['1F']['probabilities'].shape)
print('Number of labels:', len(results['1F']['true_labels']))

fig, ax = plt.subplots(figsize=(15,15))
ax.scatter(list(range(len(time_distances))), time_distances)
ax.set_ylabel('time_distances', fontsize=30)
ax.set_xlabel('utterance_ids', fontsize=30)
ax.tick_params(axis='both', which='major', labelsize=20)
ax.set_title('Time distance plots for all utterances in step-1 forecasting', fontsize=25, color='r')
plt.show()
