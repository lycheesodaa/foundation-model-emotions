import os.path
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.metrics import confusion_matrix


def _validate_features(features: Optional[NDArray]) -> bool:
    """
    Validate features array for null values and proper structure.
    """
    if features is None:
        return False

    # Check for NaN or infinite values
    if not np.all(np.isfinite(features)):
        return False

    # Check for empty arrays
    if features.size == 0:
        return False

    return True


def check_null_values(df: pd.DataFrame, speaker_id: str) -> None:
    """
    Check for null values in the DataFrame and print detailed information about any nulls found.
    Raises AssertionError with detailed information if nulls are found.
    """
    # Check each column for null values
    null_columns = df.columns[df.isnull().any()].tolist()

    if null_columns:
        error_message = f"\nNull values found in {speaker_id} data:\n"

        for col in null_columns:
            # Get indices of null values in this column
            null_indices = df[df[col].isnull()].index.tolist()
            null_rows = df.iloc[null_indices]

            error_message += f"\nColumn '{col}' has {len(null_indices)} null values:"
            error_message += f"\nNull indices: {null_indices}"
            error_message += "\nAffected rows:"
            error_message += f"\n{null_rows}\n"

            # If the column contains nested data (like features), provide more detail
            if col == 'features':
                error_message += "\nFeature shapes for null rows:"
                for idx in null_indices:
                    feat = df.loc[idx, 'features']
                    shape = feat.shape if isinstance(feat, np.ndarray) else 'Not an array'
                    error_message += f"\nIndex {idx}: {shape}"

        raise AssertionError(error_message)


def confusion_matrix_plot(speaker, y_true, y_pred, classes,
                          savedir='Images',
                          normalize=False,
                          title=None,
                          cmap=plt.colormaps.get_cmap('Blues')):
    """
    This function will create a plot of confusion matrix and also show of per class performance
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix for speaker {}'.format(speaker)
        else:
            title = 'Confusion matrix, without normalization for speaker {}'.format(speaker)
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = np.around(cm, 2)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Forecasted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()

    if not os.path.exists(savedir):
        os.mkdir(savedir)
    fig.savefig(f'{savedir}/Ses{speaker}.png', bbox_inches='tight')

    return ax


def prepare_data(directory, feature_type="dynamic", mean_only=False, verbose=False):
    print(f'Preparing data from {directory}...')

    features = {}
    labels = []
    speakers = []
    for speaker in range(1, 6):
        for gender in ['F', 'M']:
            speaker_id = f"{speaker}{gender}"

            filepath = Path(directory) / feature_type / f"audio_visual_speaker_{speaker_id}.csv"
            feature_set = pd.read_pickle(filepath)
            features[speaker_id] = np.vstack([
                feature_set["features"][idx] for idx in range(feature_set.shape[0])
            ])
            labels.extend([
                feature_set["UF_label"][idx] for idx in range(feature_set.shape[0])
            ])
            speakers.extend([speaker_id] * feature_set.shape[0])

    full_feature_set = np.vstack(list(features.values()))

    data_instances = len(labels)
    sequence_length = int(full_feature_set.shape[0] / data_instances)
    feature_dimension = 179 if mean_only else 895

    if feature_type == "dynamic":
        full_feature_set = full_feature_set.reshape(
            data_instances, sequence_length, feature_dimension
        )

    if verbose:
        print('\nTotal label counts:')
        temp = pd.DataFrame(labels, columns=['labels'])
        temp = pd.concat([temp, pd.DataFrame(speakers, columns=['speakers'])], axis=1)
        print(temp.groupby('speakers')['labels'].value_counts())

    return full_feature_set, np.asarray(labels), np.asarray(speakers)
