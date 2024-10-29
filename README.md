**This repository contains code heavily adapted from [Sadat Shahriar's Emotion Forecasting Project](https://github.com/sadat1971/Emotion-Forecasting).**

The code was optimized and adapted to fit more recent versions of pandas, and extended to process data into the format required for processing by the Moirai time-series foundation model. 

The process flow is as follows:
- Download the IEMOCAP dataset from [USC SAIL's website](https://sail.usc.edu/iemocap/iemocap_release.htm) and extract it into the `Processed/` folder. 
  - Separate the audio `.wav` files into the `Processed/Female` and `Processed/Male` folders accordingly (e.g. `Ses01F_impro01_*.wav` files for females, and `Ses01M_impro01_*.wav` files for males).
- Run the `audio_feat_extract.praat` file with [Praat](https://www.fon.hum.uva.nl/praat/).
- Run the `combine_audiovisual_data.py`, `window_based_reformation.py`, `utterance_prep.py` and `run_algorithms_pytorch.py` files, in that order.
---
# Emotion-Forecasting
**This respository contains code and step-by-step guide for completing the Emotion Forecasting Project.**
## Background:
Emotion forecasting is the task of predicting the future emotion of a speaker-i.e., the emotion label of the future speaking turn-based on the speaker's past and current audiovisual cues. Emotion forecasting systems require new problem formulations that differ from traditional emotion recognition systems.

1. Emotion Forecasting is different from emotion recognition problem. Emotion recognition analyze the current hehavioral cues and then predicts the current emotion (either classify or do regression). Emotion forecasting analyze the current data and predicts the __future__ emotion in a conversation. 
2. We fromulate the emotion forecasting problem from IEMOCAP database [1]. As the problem involves dydaic conversation, we define conversational turn or *utterance* as uninterrupted speech from one speaker. We want to forecast 1, 2 and 3 speaking turn ahead. 
We have two important hypothesis, that we test through the experiments: 
a. Sequential features work better than static features.
b. If along with current features, the features from previous history utterance is taken, forecasting performance will be better.
 
3. For hypothesis (a), We use Fully-Connected Deep Neural Network (FC-DNN) for analyzing with static feature set as **baseline** model to compare the static vs dynamic modeling. We use Long Short-Term Memory (LSTM) and Bidirectional Long Short-Term Memory (BLSTM) to compare the performance. For hypothesis (b), we use LSTM and BLSTM __without__ history as baseline. Then we add history to our featureset and perform forecasting with __added__ history. 
4. Our experimental results on the IEMOCAP benchmark dataset demonstrate that BLSTM and LSTM outperform FC-DNN by up to 2.42% in unweighted recall. When using both the current and past utterances, deep dynamic models show an improvement of up to 2.39% compared to their performance when using only the current utterance.

**The following table gives an overview of our work**


| Topic         | Information |
| ------------- | ------------- |
| # of data     | 2823 for 1 step Utterance Forecasting, 2734 for 2 step, and 2662 for 3 step  |
| # of features  | Audio: 205, Visual (facial): 690, total __895__ features  |
| # The Dataset | IEMOCAP, with 10 participant subjects. Total 12 hour of conversation |
| Machine Learning Models | FC-DNN, LSTM, BLSTM |
| Approach  | Classification |
| # of classes | Angry, Happy, Neutral, Sad (__4 classes__) |
| Major libraries for data analysis | Numpy, Scipy, Pandas, re |
| ML libraries | Keras, Scikit-learn |

**Below is the code structure for the project**

| Code file | Task |
| ------------- | ------------- |
| audio_feat_extract.praat    | Extract audio features from raw ```.wav``` files |
|  Combine_audiovisual_data.py| Combine the audio-visual information, filling out the missing values and clean the data for feature engineering. Produce a table for each participant (speaker) at the end  |
| window_based_reformation.py | Produce statistical features from raw audio-visual information. Doing necessary feature engineering |
| Utt_Fore_Data_Prep.py   | Preparing the dataset for different step utetrance forecasting. Also prepares the dataset for both history-less and history-added version of emotion forecasting. Expplained in ![History-Less Emotion Forecasting](images/cur.pdf) and [History-Added emotion forecasting](/images/his.pdf) |
| run_algorithms.py | Contains classes to run FC-DNN, LSTM and BLSTM |
|  	Main_Process_with_visualization.ipynb | Provides details of how to work on a specific cases of emotion forecasting. Contains visualization of result and time-impact of forecasting |
| Performance_Comparison_Visualization.ipynb | Shows the comparison of performance by testing our two hypotheis |




For more details, please refer to our work [here](https://ieeexplore.ieee.org/document/8756599)




## References:
[1] C. Busso, M. Bulut, C.C. Lee, A. Kazemzadeh, E. Mower, S. Kim, J.N. Chang, S. Lee, and S.S. Narayanan, "IEMOCAP: Interactive emotional dyadic motion capture database," Journal of Language Resources and Evaluation, vol. 42, no. 4, pp. 335-359, December 2008.
