import pandas as pd
from scipy import interpolate
import numpy as np
import re
import os
from pathlib import Path
from window_based_reformation import check_null_values

# numbers represent LLID, RLID, MH, MNOSE, LNSTRL, TNOSE, RNSTRL, LHD and RHD respectively
excluded_video_feature_nums = [23, 24, 25, 26, 27, 28, 29, 60, 61]
excluded_video_features = []
for num in excluded_video_feature_nums:
    excluded_video_features.extend([f"X{num}", f"Y{num}", f"Z{num}"])


def interpolate_column(column, target_length):
    # Create original indices
    original_indices = np.arange(len(column))

    # Create target indices for interpolation
    target_indices = np.linspace(0, len(column) - 1, target_length)

    # Create interpolation function
    f = interpolate.interp1d(
        original_indices, column, kind="linear", fill_value="extrapolate"
    )

    # Generate interpolated values
    interpolated = f(target_indices)

    return interpolated


def downsample_data_simple(df, value):
    """
    Downsample DataFrame by keeping every nth row (where n = value)
    """
    return df.iloc[::value]


def downsample_data(data, value):
    del_row = [i for i in range(len(data)) if i % value != 0]
    new_data = np.delete(data, del_row, axis=0)
    return new_data


def fill_missing_value_simple(df):
    """
    Simpler version using pandas built-in interpolation methods.
    This will do interpolation along each column independently.
    """
    result_df = df.interpolate(method="cubic", limit_direction="both")    
    
    # Check for remaining NaN values (can occur at edges or with sparse data)
    if result_df.isna().any().any():
        result_df = result_df.ffill().bfill()
    
    # Final check for any remaining NaN values
    if result_df.isna().any().any():
        raise RuntimeError("Failed to fill all missing values. Please try a different fallback_method.")
    
    return result_df


def fill_missing_values_df(df, threshold=0.3, fallback_method='ffill'):
    """
    Fill missing values in a DataFrame using interpolation.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame containing missing values
    threshold : float, default=0.3
        Maximum allowed proportion of missing values (30% by default)
    fallback_method : str, default='ffill'
        Method to use for filling any remaining missing values after interpolation
        Options: 'ffill', 'bfill', 'mean', 'median'
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with all missing values filled
        
    Raises:
    -------
    ValueError
        If the proportion of missing values exceeds the threshold
    """
    # Check if input is a DataFrame
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    # Create a copy to avoid modifying the original DataFrame
    df_copy = df.copy()
    
    # Calculate proportion of missing values
    missing_prop = df_copy.isna().sum().sum() / (df_copy.shape[0] * df_copy.shape[1])
    
    if missing_prop > threshold:
        raise ValueError(f"Missing value proportion ({missing_prop:.2%}) exceeds threshold ({threshold:.2%})")
    
    # Convert DataFrame to numpy array for interpolation
    matrix = df_copy.values.astype(float)
    
    # Create coordinate grid
    x = np.arange(0, matrix.shape[1])
    y = np.arange(0, matrix.shape[0])
    xx, yy = np.meshgrid(x, y)
    
    # Mask invalid values
    masked_matrix = np.ma.masked_invalid(matrix)
    
    # Get valid coordinates and values
    x1 = xx[~masked_matrix.mask]
    y1 = yy[~masked_matrix.mask]
    valid_values = matrix[~masked_matrix.mask]
    
    # Perform cubic interpolation
    filled_matrix = interpolate.griddata(
        (x1, y1), valid_values.ravel(), (xx, yy), method='cubic'
    )
    
    # Convert back to DataFrame
    result_df = pd.DataFrame(filled_matrix, index=df_copy.index, columns=df_copy.columns)
    
    # Check for remaining NaN values (can occur at edges or with sparse data)
    if result_df.isna().any().any():
        if fallback_method == 'ffill':
            result_df = result_df.ffill().bfill()
        elif fallback_method == 'bfill':
            result_df = result_df.bfill().ffill()
        elif fallback_method == 'mean':
            result_df = result_df.fillna(result_df.mean())
        elif fallback_method == 'median':
            result_df = result_df.fillna(result_df.median())
        else:
            raise ValueError("Invalid fallback_method. Choose from: 'ffill', 'bfill', 'mean', 'median'")
    
    # Final check for any remaining NaN values
    if result_df.isna().any().any():
        raise RuntimeError("Failed to fill all missing values. Please try a different fallback_method.")
        
    return result_df


def fill_missing_values(matrix):
    """
    This function will fill the missing nan values in a matrix.
    It will be used when we find less than 30% missing value
    """
    x = np.arange(0, matrix.shape[1])
    y = np.arange(0, matrix.shape[0])
    # mask invalid values
    matrix = np.ma.masked_invalid(matrix)
    xx, yy = np.meshgrid(x, y)
    # get only the valid values
    x1 = xx[~matrix.mask]
    y1 = yy[~matrix.mask]
    newarr = matrix[~matrix.mask]

    filled_matrix = interpolate.griddata(
        (x1, y1), newarr.ravel(), (xx, yy), method="cubic"
    )
    return filled_matrix


class CombiningAV:
    """
    This class works for combining the audio and visual data of IEMOCAP dataset
    """

    def __init__(self, audio_feature_location, video_feature_location, output_dir):
        self.audio_feature_location = audio_feature_location
        self.video_feature_location = video_feature_location
        self.output_dir = output_dir
        print("Combining the feature set...")

    def read_mocap_file(self, session, session_filename):
        """
        Read motion capture data from a text file into a pandas DataFrame.
        The file has a two-row header structure:
        - First row: Frame#, Time and all facial labels
        - Second row: Three dimensional column names e.g. (X01, Y01, Z01) for CH1

        Parameters:
        session (int): Session number
        session_filename (str): Name of session file without the extension

        Returns:
        pandas.DataFrame: DataFrame containing the motion capture data
        """
        conversation_prefix = session_filename.rsplit("_", 1)[0]
        filepath = (
            Path(self.video_feature_location)
            / f"Session{session}/sentences/MOCAP_rotated"
            / conversation_prefix
            / f"{session_filename}.txt"
        )

        # Read all lines from the file
        with open(filepath, "r") as file:
            lines = [line.strip() for line in file if line.strip()]

        # Get the two header rows
        header_row1 = lines[0].split()  # First row with Frame# and Time
        header_row2 = lines[1].split()  # Second row with additional column names

        # Combine the headers
        # Take 'Frame#' and 'Time' from first row, and the rest from second row
        columns = ["Frame#", "Time"] + header_row2

        # Get the data lines (everything after the headers that starts with a number)
        data_lines = [line for line in lines[2:] if line[0].isdigit()]

        # Parse the data lines into a list of lists
        data = [line.split() for line in data_lines]

        # Create DataFrame
        df = pd.DataFrame(data, columns=columns)
        df = df[[col for col in df.columns if col not in excluded_video_features]]

        assert len(df.columns) == (
            46 * 3 + 2
        )  # 46 facial features * 3 dimensions, and Frame# and Time

        # Convert numeric columns to float
        numeric_columns = df.columns[
            df.columns != "Frame#"
        ]  # All columns except Frame#
        df[numeric_columns] = df[numeric_columns].astype(float)
        df["Frame#"] = df["Frame#"].astype(int)

        return df

    def load_audio_files(self, session, name):

        # Pitch
        pitch = os.path.join(self.audio_feature_location, "{}.pitch".format(name))
        P = pd.read_csv(pitch, header=None)[0].to_list()
        P = [float(value) for count, value in enumerate(P) if count >= 4]

        # Energy or intensity
        energy = os.path.join(
            self.audio_feature_location,
            "{}.intensity".format(name),
        )
        E = pd.read_csv(energy, header=None)[0].to_list()
        E = [float(value) for count, value in enumerate(E) if count >= 4]

        # MFb
        mfb = os.path.join(self.audio_feature_location, "{}.mfb".format(name))
        MFB = pd.read_csv(mfb, header=None)[0].to_list()
        List_MFB_string = [
            re.findall(r"[+-]?\d+\.?\d*(?:[eE][+-]?\d+)?", i) for i in MFB
        ]
        MFB_value = []
        for l in List_MFB_string:
            MFB_value.append([float(i) for i in l])

        # MFCC
        mfcc = os.path.join(self.audio_feature_location, "{}.mfcc".format(name))
        MFCC = pd.read_csv(mfcc, header=None)[0].to_list()
        List_MFCC_string = [
            re.findall(r"[+-]?\d+\.?\d*(?:[eE][+-]?\d+)?", i) for i in MFCC
        ]
        MFCC_value = []
        for l in List_MFCC_string:
            MFCC_value.append([float(i) for i in l])

        # First four rows of MFB and MFCC arrays are col names etc.
        # MFCC values have 13 columns, as the first feature contains the C0 value, related to the energy, and is not contained in the number_of_coefficients=12 argument
        return P, E, MFB_value[4:], [row[1:] for row in MFCC_value[4:]]

    def produce_speakerwise_AV_data(self):
        """
        This code will produce speaker-wise audio-visual data with a same framerate.
        Also, it will operate on the NaN features and remove or replace them.

        For the old code:
        v_data contains 2(?) dimensions
        Only the first item in the first dimension i.e. v_data[0] is relevant
        Second dimension contains a vector of dictionaries including:
            - name: list of filenames(?)
            - video: video features
            - categorical: label
        """

        audio_file_list = os.listdir(self.audio_feature_location)
        audio_file_list = [file.split(".")[0] for file in audio_file_list]
        audio_file_list = list(set(audio_file_list))
        audio_file_list.sort()
        print("Total number of audio files:", len(audio_file_list))

        for session in range(1, 6):
            for gender in ["F", "M"]:
                audio_visual_df = pd.DataFrame(
                    columns=["name", "video", "audio"]
                )

                speaker_prefix = f"Ses0{session}{gender}"
                speaker_file_names = [
                    file for file in audio_file_list if file.startswith(speaker_prefix)
                ]
                # speaker_file_names = ['Ses03F_script01_1_M037']

                for speaker_file_name in speaker_file_names:
                    # Load audio features
                    Pitch, Energy, MFB, MFCC = self.load_audio_files(
                        session, speaker_file_name
                    )
                    # print(len(Pitch))
                    # print(len(Energy))
                    # print(len(MFB), len(MFB[1]))
                    # print(len(MFCC), len(MFCC[1]))

                    # Load video features
                    try:
                        video_features = self.read_mocap_file(
                            session, speaker_file_name
                        )
                    except FileNotFoundError as e:
                        print(e)
                        continue
                    # print(video_features.shape)

                    # Check the percentage of nan values in the video features
                    percent_nans = video_features.isnull().mean().max()

                    if any([P > 0 for P in Pitch]) and percent_nans < 0.3:
                        # Our main focus is pitch. For the start and end zeros of pitch
                        # means, there are no information.

                        non_zero_pitch_idx = [
                            loc for loc, val in enumerate(Pitch) if val != 0
                        ]
                        start = min(non_zero_pitch_idx)

                        finish = min(
                            [
                                max(non_zero_pitch_idx) + 1,
                                len(Energy),
                                len(MFB),
                                len(MFCC),
                            ]
                        )
                        Length = finish - start

                        downsampled_video = downsample_data_simple(
                            video_features, 3
                        )

                        nan_removed_video = fill_missing_values_df(
                            downsampled_video
                        )
                        # print(nan_removed_video)

                        check_null_values(nan_removed_video, speaker_file_name)

                        try:
                            audio_data_matrix = np.concatenate(
                                (
                                    np.asarray(Pitch[start:finish]).reshape(1, Length),
                                    np.asarray(Energy[start:finish]).reshape(1, Length),
                                    np.transpose(np.asarray(MFB[start:finish])),
                                    np.transpose(np.asarray(MFCC[start:finish])),
                                ),
                                axis=0,
                            )
                        except Exception as e:
                            target_length = len(MFB[0])
                            for i, row in enumerate(MFB[start:finish]):
                                if len(row) != target_length:
                                    print(i)
                                    print(row)
                                    MFB[i] = interpolate_column(
                                        MFB[i], target_length
                                    )

                            target_length = len(MFCC[0])
                            for i, row in enumerate(MFCC[start:finish]):
                                if len(row) != target_length:
                                    print(i)
                                    print(row)
                                    MFCC[i] = interpolate_column(
                                        MFCC[i], target_length
                                    )

                        audio_data_matrix = np.transpose(audio_data_matrix)

                        assert len(audio_data_matrix) == len(
                            nan_removed_video[start:finish]
                        )

                        av_data = {
                            "name": speaker_file_name,
                            "video": nan_removed_video[start:finish],
                            "audio": audio_data_matrix,
                            # "label": v_data[0][data]["categorical"][0][0],
                        }
                        audio_visual_df = pd.concat(
                            [audio_visual_df, pd.DataFrame.from_dict([av_data])],
                            ignore_index=True,
                        )

                        print(
                            f"Processed up to speaker {session}{gender} and utterance {speaker_file_name}."
                        )
                
                if not os.path.exists(self.output_dir):
                    os.mkdir(self.output_dir)

                audio_visual_df = audio_visual_df.merge(
                    pd.read_csv("Processed/clustered_labels.csv"),
                    left_on="name",
                    right_on="utterance",
                    how="left",
                ).drop('utterance', axis=1)
                audio_visual_df.to_pickle(
                    f"{self.output_dir}/audio_visual_speaker_{session}{gender}.csv"
                )
                print(
                    f"Exported to: {self.output_dir}/audio_visual_speaker_{session}{gender}.csv\n"
                )


if __name__ == "__main__":
    #### Main function code testing ####
    Data = CombiningAV("Processed/Features_50_25/", "Processed/IEMOCAP_full_release/", "Files/sameframe_50_25")
    Data.produce_speakerwise_AV_data()
