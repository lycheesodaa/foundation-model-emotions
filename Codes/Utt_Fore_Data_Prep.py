#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 12:50:55 2019

@author: sadat

[This file is not in use]
"""
import re
import pandas as pd
import os
import numpy as np


class Prepare_UF_Cur_Data:
    """
    This class will prepare the Utterance Forecasting data both for UF-cur
    and UF-his approach
    """

    def __init__(self):
        print("Creating dataset for utterance forecasting...")

    def organize_IEMOCAP_info(self, info_file="IEMOCAP_EmoEvaluation.txt"):
        """
        This function will parse the timing information from
        IEMOCAP_EmoEvaluation.txt and prepare a dataframe like a look-up-table
        """
        directory = os.path.join("Files", info_file)
        IEMOCAP_info = pd.read_csv(directory, sep="\n")
        organized_IEMOCAP = pd.DataFrame(
            columns=["utterance_name", "start_time", "end_time"]
        )

        for i in IEMOCAP_info.index:
            value = IEMOCAP_info[
                "% [START_TIME - END_TIME] TURN_NAME EMOTION [V, A, D]"
            ][i]
            if "Ses" in value:
                field_splitted = value.split("\t")
                timing = re.findall(r"\d+\.\d+", field_splitted[0])
                utterance_info = {
                    "utterance_name": field_splitted[1],
                    "start_time": float(timing[0]),
                    "end_time": float(timing[1]),
                }

                organized_IEMOCAP = organized_IEMOCAP.append(
                    utterance_info, ignore_index=True
                )
        return organized_IEMOCAP

    def find_time_distance(self, look_up_table, cur_utt, future_utt):
        """
        This function will find the time-distance between current utterance and
        the utterance to be forecasted.
        """
        cur_utt_starts = look_up_table[
            look_up_table["utterance_name"].str.contains(cur_utt)
        ]["start_time"].tolist()[0]

        cur_utt_ends = look_up_table[
            look_up_table["utterance_name"].str.contains(cur_utt)
        ]["end_time"].tolist()[0]
        future_utt_starts = look_up_table[
            look_up_table["utterance_name"].str.contains(future_utt)
        ]["start_time"].tolist()[0]

        future_utt_ends = look_up_table[
            look_up_table["utterance_name"].str.contains(future_utt)
        ]["end_time"].tolist()[0]

        return (future_utt_ends + future_utt_starts) / 2 - (
            cur_utt_ends + cur_utt_starts
        ) / 2

    def do_zero_pad_and_save(
        self,
        speakerwise_data_dict,
        Utt_Length,
        step,
        forecast_window,
        context,
        feature_type="dynamic",
    ):
        # This function will do zero-padding for dynamic or sequential dataset and save them. If
        # processing the static dataset, no zero-padding is required !
        for speaker in range(1, 6):
            for gender in ["F", "M"]:
                UF_data = speakerwise_data_dict[f"{speaker}{gender}"]
                if feature_type == "dynamic":
                    for idx in range(UF_data.shape[0]):
                        UF_data["features"][idx] = np.pad(
                            UF_data["features"][idx],
                            (
                                (
                                    0,
                                    max(Utt_Length) - UF_data["features"][idx].shape[0],
                                ),
                                (0, 0),
                            ),
                            "constant",
                            constant_values=0,
                        )

                output_dir = (
                    f"Files/{forecast_window}_{context}_data/step_{step}/{feature_type}"
                )
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                UF_data.to_pickle(
                    f"{output_dir}/audio_visual_speaker_{speaker}{gender}.csv"
                )

    def creating_dataset(self, step=0, feature_type="dynamic", normalization=False):
        """This function creates the UF data. If feature_type is set to
        'static', it will create static data."""
        # This dict will save the dataframe for different speaker's processed data
        speakerwise_data = {}
        # This will record the length of utterances to use for zero padding
        Utt_Length = []
        look_up_table = self.organize_IEMOCAP_info()
        for speaker in range(1, 6):
            for gender in ["F", "M"]:
                UF_data = pd.DataFrame(
                    columns=[
                        "cuurent_name",
                        "features",
                        "UF_label",
                        "forecasted_utt_name",
                        "time_distance",
                    ]
                )

                # First, let's compare the current utterance and to-be-forecasted
                # utterance. If they are of same speaker and same dialog, record
                # That. Otherwise, ignore.
                file = os.path.join(
                    "Files",
                    "statistical",
                    feature_type,
                    f"audio_visual_speaker_{speaker}{gender}.csv",
                )
                audio_visual_data = pd.read_pickle(file)
                # remember, the last utterance will not have any labels. This is
                # the reason, we can use utterances upto
                # total_utterance - forecasting_step
                for data_idx in range(len(audio_visual_data) - step):
                    # First, let's compare the current utterance and to-be-forecasted
                    # utterance. If they are of same speaker and same dialog, record
                    # that. Otherwise, ignore.
                    current_utt_name = audio_visual_data["name"][data_idx]
                    target_utt_name = audio_visual_data["name"][data_idx + step]

                    if (
                        current_utt_name[:-5] == target_utt_name[:-5]
                        and audio_visual_data["label"][data_idx + step] < 4
                    ):
                        Utt_Length.append(
                            audio_visual_data["stat_features"][data_idx].shape[0]
                        )
                        data_instance = {
                            "cuurent_name": current_utt_name,
                            "features": audio_visual_data["stat_features"][data_idx],
                            "UF_label": audio_visual_data["label"][data_idx + step],
                            "forecasted_utt_name": target_utt_name,
                            "time_distance": self.find_time_distance(
                                look_up_table, current_utt_name, target_utt_name
                            ),
                        }
                        UF_data = UF_data.append(data_instance, ignore_index=True)
                        print(f"speaker is {speaker}{gender} and data is {data_idx}")

                if normalization:
                    All_Feat = np.vstack(
                        ([UF_data["features"][i] for i in range(UF_data.shape[0])])
                    )
                    Mean_Feat = np.mean(All_Feat, axis=0).reshape(1, All_Feat.shape[1])
                    Std_Feat = np.std(All_Feat, axis=0).reshape(1, All_Feat.shape[1])

                    for idx in range(UF_data.shape[0]):
                        UF_data["features"][idx] = (
                            UF_data["features"][idx] - Mean_Feat
                        ) / Std_Feat
                    print("Normalizing Done")

                speakerwise_data[f"{speaker}{gender}"] = UF_data

        self.do_zero_pad_and_save(
            speakerwise_data, Utt_Length, step, "UF", "Cur", feature_type
        )


class Prepare_UF_history_Data(Prepare_UF_Cur_Data):
    """
    This class will create UF data with added history-context. It inherits the
    Prepare_UF_Cur_Data class.
    """

    def __init__(self):
        pass

    def creating_dataset(self, step=0, normalization=False):
        speakerwise_data = {}
        # This will record the length of utterances to use for zero padding
        Utt_Length = []
        look_up_table = self.organize_IEMOCAP_info()

        for speaker in range(1, 6):
            for gender in ["F", "M"]:
                UF_data = pd.DataFrame(
                    columns=[
                        "cuurent_name",
                        "features",
                        "UF_label",
                        "forecasted_utt_name",
                        "time_distance",
                    ]
                )
                file = os.path.join(
                    "Files",
                    "statistical",
                    f"audio_visual_speaker_{speaker}{gender}.csv",
                )
                audio_visual_data = pd.read_pickle(file)

                for data_idx in range(len(audio_visual_data) - step):
                    current_utt_name = audio_visual_data["name"][data_idx]
                    target_utt_name = audio_visual_data["name"][data_idx + step]
                    try:
                        history_utt_name = audio_visual_data["name"][data_idx - 1]
                        if current_utt_name[:-5] == history_utt_name[:-5]:
                            if (
                                current_utt_name[:-5] == target_utt_name[:-5]
                                and audio_visual_data["label"][data_idx + step] < 4
                            ):
                                data_instance = {
                                    "cuurent_name": current_utt_name,
                                    "features": np.vstack(
                                        (
                                            audio_visual_data["stat_features"][
                                                data_idx - 1
                                            ],
                                            audio_visual_data["stat_features"][
                                                data_idx
                                            ],
                                        )
                                    ),
                                    "UF_label": audio_visual_data["label"][
                                        data_idx + step
                                    ],
                                    "forecasted_utt_name": target_utt_name,
                                    "time_distance": self.find_time_distance(
                                        look_up_table, current_utt_name, target_utt_name
                                    ),
                                }
                                UF_data = UF_data.append(
                                    data_instance, ignore_index=True
                                )
                                print(
                                    f"speaker is {speaker}{gender} and data is {data_idx}"
                                )
                                Utt_Length.append(data_instance["features"].shape[0])

                            else:
                                pass
                        else:
                            data_instance = {
                                "cuurent_name": current_utt_name,
                                "features": audio_visual_data["stat_features"][data_idx],
                                "UF_label": audio_visual_data["label"][data_idx + step],
                                "forecasted_utt_name": target_utt_name,
                                "time_distance": self.find_time_distance(
                                    look_up_table, current_utt_name, target_utt_name
                                ),
                            }
                            UF_data = UF_data.append(data_instance, ignore_index=True)
                            Utt_Length.append(data_instance["features"].shape[0])
                    except:
                        # This except block will work only for the first utterance
                        data_instance = {
                            "cuurent_name": current_utt_name,
                            "features": audio_visual_data["stat_features"][data_idx],
                            "UF_label": audio_visual_data["label"][data_idx + step],
                            "forecasted_utt_name": target_utt_name,
                            "time_distance": self.find_time_distance(
                                look_up_table, current_utt_name, target_utt_name
                            ),
                        }
                        UF_data = UF_data.append(data_instance, ignore_index=True)
                        Utt_Length.append(data_instance["features"].shape[0])

                if normalization:
                    All_Feat = np.vstack(
                        ([UF_data["features"][i] for i in range(UF_data.shape[0])])
                    )
                    Mean_Feat = np.mean(All_Feat, axis=0).reshape(1, All_Feat.shape[1])
                    Std_Feat = np.std(All_Feat, axis=0).reshape(1, All_Feat.shape[1])

                    for idx in range(UF_data.shape[0]):
                        UF_data["features"][idx] = (
                            UF_data["features"][idx] - Mean_Feat
                        ) / Std_Feat
                    print("Normalizing Done")

                speakerwise_data[f"{speaker}{gender}"] = UF_data

        self.do_zero_pad_and_save(speakerwise_data, Utt_Length, step, "UF", "His")


# UF_cur = Prepare_UF_Cur_Data()
# UF_cur.creating_dataset(step=1, feature_type='dynamic', normalization=True)
UF_his = Prepare_UF_history_Data()
UF_his.creating_dataset(step=1, feature_type="dynamic", normalization=True)
