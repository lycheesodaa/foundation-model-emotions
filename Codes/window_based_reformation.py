import pandas as pd
import os
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
from numpy.typing import NDArray


def make_window_idx(frame_length, frame_idx, overlap, window_type):
    """
    This function will create the index list for making overlapped windows,
    through which, we will produce statistical features.

    For example, make_window_idx(80, 30, 15, window_type='dynamic') returns:
    [[0, 29], [15, 44], [30, 59], [45, 74], [60, 79]]

    If window_type='static', it will return:
    [[0, 79]]
    """
    index_list = []

    if window_type == "dynamic":
        i = 0
        j = 0
        while j < frame_length:
            if (i + frame_idx - 1) < frame_length:
                index_list.append([i, i + frame_idx])
                j = i + frame_idx - 1
            else:
                index_list.append([i, frame_length])
                break
            i += overlap
    elif window_type == "static":
        index_list = [[0, frame_length]]

    return index_list


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


class WindowBasedReformation:
    """
    This code converts the frame level data into a window-based reformed data.
    It also produces the statistical features for those windows
    """

    def __init__(self, file_location):
        self.file_location = file_location
        print("Creating window based features...")

    def process_data(self, window_type):
        """
        Function which creates the statistical window-based data
        """

        for speaker in range(1, 6):
            for gender in ["F", "M"]:
                filename = f"audio_visual_speaker_{speaker}{gender}.csv"
                # filename = f"audio_visual_speaker_3F.csv"

                audio_visual_df = pd.DataFrame(
                    columns=["name", "stat_features", "label"]
                )
                directory = os.path.join(self.file_location, filename)
                audio_visual_framewise = pd.read_pickle(directory)
                assert len(audio_visual_framewise.columns) == 4
                # print(audio_visual_framewise['audio'])
                # print(audio_visual_framewise['video'])
                # print(audio_visual_framewise.columns)
                # exit()

                for utterance in range(len(audio_visual_framewise)):
                    index_list = make_window_idx(
                        audio_visual_framewise["audio"][utterance].shape[0],
                        30,
                        15,
                        window_type,
                    )
                    # the first two columns of 'video' are Frame# and Time, exclude them
                    features_concatenated = np.concatenate(
                        (
                            audio_visual_framewise["audio"][utterance],
                            audio_visual_framewise["video"][utterance].iloc[:, 2:],
                        ),
                        axis=1,
                    )

                    # print(audio_visual_framewise['audio'][utterance])
                    # print(len(audio_visual_framewise['audio'][utterance]), len(audio_visual_framewise['audio'][utterance][1]))
                    # print()
                    # print(audio_visual_framewise['video'][utterance])
                    # print(len(audio_visual_framewise['video'][utterance]))
                    # print()
                    # print(features_concatenated.shape)
                    # exit()

                    # ???
                    # if utterance == 196:
                    #     print(features_concatenated)

                    # Extract statistical information from window-based data
                    window_wise_feature = np.zeros(
                        (len(index_list), 895)
                    )  # 5 statistical features from each of the 179 features.
                    for idx in range(len(index_list)):
                        parsed_features = features_concatenated[
                            index_list[idx][0] : index_list[idx][1]
                        ]
                        statistical_feat = np.concatenate(
                            (
                                np.mean(parsed_features, axis=0).reshape(1, 179),
                                np.std(parsed_features, axis=0).reshape(1, 179),
                                np.quantile(parsed_features, 0.25, axis=0).reshape(
                                    1, 179
                                ),
                                np.quantile(parsed_features, 0.75, axis=0).reshape(
                                    1, 179
                                ),
                                np.quantile(parsed_features, 0.75, axis=0).reshape(
                                    1, 179
                                )
                                - np.quantile(parsed_features, 0.25, axis=0).reshape(
                                    1, 179
                                ),
                            ),
                            axis=1,
                        )

                        assert _validate_features(statistical_feat),\
                            (f'statistical feature vector for {speaker}{gender} - utt{utterance}, window{idx} has nulls\n'
                             f'{statistical_feat}')

                        window_wise_feature[idx, :] = statistical_feat
                        print(
                            f"speaker is {speaker}{gender} and data is {utterance}, len of the idx is {len(index_list)}"
                        )
                    feature_set = {
                        "name": audio_visual_framewise["name"][utterance],
                        "stat_features": window_wise_feature,
                        "label": audio_visual_framewise["label"][utterance],
                    }
                    audio_visual_df = pd.concat(
                        [audio_visual_df, pd.DataFrame.from_dict([feature_set])],
                        ignore_index=True,
                    )

                output_dir = f"Files/statistical/{window_type}/"

                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                audio_visual_df.to_pickle(output_dir + filename)
                print(f"Exported to: {output_dir}{filename}")


if __name__ == "__main__":
    # Main function for test only
    
    task = WindowBasedReformation("Files/sameframe_50_25")
    # task.process_data("static")
    task.process_data("dynamic")
