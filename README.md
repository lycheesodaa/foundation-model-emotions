# Emotion-Forecasting
**This respository contains code and step-by-step guide for completing the Emotion Forecasting Project.**

Emotion forecasting is the task of predicting the future emotion of a speaker-i.e., the emotion label of the future speaking turn-based on the speaker's past and current audiovisual cues. Emotion forecasting systems require new problem formulations that differ from traditional emotion recognition systems.

1. Emotion Forecasting is different from emotion recognition problem. Emotion recognition analyze the current hehavioral cues and then predicts the current emotion (either classify or do regression). Emotion forecasting analyze the current data and predicts the __future__ emotion in a conversation. 
2. We fromulate the emotion forecasting problem from IEMOCAP database [1]. As the problem involves dydaic conversation, we define conversational turn or *utterance* as uninterrupted speech from one speaker. We want to forecast 1, 2 and 3 speaking turn ahead. 
We have two important hypothesis, that we test through the experiments: 
a. Sequential features work better than static features.
b. If along with current features, the features from previous history utterance is taken, forecasting performance will be better.
3. For hypothesis (a), Ww use Fully-Connected Deep Neural Network (FC-DNN) for analyzing with static feature set as **baseline** model to compare the static vs dynamic modeling. We use Long Short-Term Memory (LSTM) and Bidirectional Long Short-Term Memory (BLSTM) to compare the performance. For hypothesis (b), we use LSTM and BLSTM __without__ history as baseline. Then we add history to our featureset and perform forecasting with __added__ history. 
4. Our experimental results on the IEMOCAP benchmark dataset demonstrate that BLSTM and LSTM outperform FC-DNN by up to 2.42% in unweighted recall. When using both the current and past utterances, deep dynamic models show an improvement of up to 2.39% compared to their performance when using only the current utterance.


## References:
[1] C. Busso, M. Bulut, C.C. Lee, A. Kazemzadeh, E. Mower, S. Kim, J.N. Chang, S. Lee, and S.S. Narayanan, "IEMOCAP: Interactive emotional dyadic motion capture database," Journal of Language Resources and Evaluation, vol. 42, no. 4, pp. 335-359, December 2008.
