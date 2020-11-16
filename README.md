#  Adversarial Learning for Fair CTC Prediction in Automatic Speech Recognition (ASR)

## The work was done in collaboration with John Harvill, UIUC (GitHub profile jharvill23). 

## Description
The script trains the CTC model based on Bi-directional LSTM encoder for ASR with demographic parity fairness constraint. The fairness is based on the adversarial classifier that tries to identify the sensitive attribute of a speaker. The models were trained and tested using the UASPEECH database of dysarthric speech.

Results demonstrate significant reduction in perfromance gap, as well as, the increase of overall accuracy in ASR task.

## How to run:
We have included the file Dysarthria_Fairness.yml to create the environment with conda.

Step 1: Run preprocessing.py. Note that you need to change config.directories.root to the location where you
store the UASPEECH corpus. Also, move the control speakers to the same directory (they are in a subdirectory called
"control", but it's easiest to have all speakers in the same directory.

Step 2: Run extract_features.py.

Step 3: Run train.py. During the import of utils.py, the files dict.pkl, wordlist.pkl, phones.pkl, and partition.pkl
will be created from dict.txt and wordlist.csv. Towards the top of train.py there are options:

TRAIN = True
EVAL = False
LOAD_MODEL = False
PERFORMANCE_EVAL = False
WORDSPLIT = False  # WORDSPLIT and UTTERANCESPLIT (one should be true and other false)
UTTERANCE_SPLIT = True

The example above will train a fresh model using the utterance split partition scheme. Currently, only the utterance split scheme is implemented.

To evaluate the model after training, you need to change TRAIN to False and EVAL, LOAD_MODEL, PERFORMANCE_EVAL to True. Also you can adjust batch size for your video RAM in
config.train.batch_size.

## Dependencies:
Python 3.6, PyTorch 1.3.1, NumPy, OS, Librosa, Tensorflow 1.13.1, Tensorboard, CUDA 10

## Authors:
  Dias Issa, PhD candidate,
  Artificial Intelligence and Machine Learning Laboratory (AIM Lab, https://slsp.kaist.ac.kr/xe/),   
  School of Electrical Engineering,  
  Korea Advanced Institute of Science and Technology (KAIST),  
  Daejeon, Korea,  
  dias.issa@kaist.ac.kr;
  
  John Harvill, PhD candidate,  
  Statistical Speech Technology lab,
  Electrical and Computer Engineering,  
  University of Illinois Urbana-Champaign, 
  Champaign, IL, USA

## Used material:
UASPEECH database: http://www.isle.illinois.edu/speech_web_lg/data/UASpeech/index.shtml

  

