import pandas as pd
import os
import numpy as np
from utils.utils import _validate_features, check_null_values

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


class WindowBasedReformation:
    """
    This code converts the frame level data into a window-based reformed data.
    It also produces the statistical features for those windows
    """

    def __init__(self, file_location):
        self.file_location = file_location
        print("Creating window based features...")

    def process_data(self, window_type, mean_only=False):
        """
        Function which creates the statistical window-based data
        """
        windows = []

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
                    windows.append(len(index_list))

                    # temporary step to ensure that the video data doesn't contain nulls
                    if audio_visual_framewise["video"][utterance].isnull().any().any():
                        audio_visual_framewise["video"][utterance] = audio_visual_framewise["video"][utterance].ffill().bfill()

                    assert audio_visual_framewise["video"][utterance].isnull().sum().sum() == 0

                    # the first two columns of 'video' are Frame# and Time, exclude them
                    features_concatenated = np.concatenate(
                        (
                            audio_visual_framewise["audio"][utterance],
                            audio_visual_framewise["video"][utterance].iloc[:, 2:],
                        ),
                        axis=1,
                    )

                    # Extract statistical information from window-based data
                    if mean_only:
                        window_wise_feature = np.zeros((len(index_list), 179))
                    else:
                        window_wise_feature = np.zeros(
                            (len(index_list), 895)
                        )  # 5 statistical features from each of the 179 features.

                    for idx in range(len(index_list)):
                        parsed_features = features_concatenated[
                            index_list[idx][0] : index_list[idx][1]
                        ]
                        if mean_only:
                            statistical_feat = np.mean(parsed_features, axis=0).reshape(1, 179)
                        else:
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

                        # assert _validate_features(statistical_feat),\
                        #     (f'statistical feature vector for {speaker}{gender} - utt{utterance}, window{idx} has nulls\n'
                        #      f'{statistical_feat}')

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

                if mean_only:
                    output_dir += 'historical_mean/'

                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                audio_visual_df.to_pickle(output_dir + filename)
                print(f"Exported to: {output_dir}{filename}")

        print('Average window length throughout all utterances:', np.mean(windows))
        print('Max window length:', np.max(windows))
        print('Min window length:', np.min(windows))
        print('Number of windows:', len(windows))


if __name__ == "__main__":
    # Main function for test only

    task = WindowBasedReformation("Files/sameframe_50_25")
    # task.process_data("static")
    task.process_data("dynamic", mean_only=True)
