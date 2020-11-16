"""
I have included the file Dysarthria_Fairness.yml if you would like to create the environment with conda. This may
save some time with installing some of the audio libraries used in preprocessing.

Step 1: Run preprocessing.py. Note that you need to change config.directories.root to the location where you
store the UASPEECH corpus. Also, move the control speakers to the same directory (they are in a subdirectory called
"control", but it's easiest to have all speakers in the same directory. There are about 140k files so it takes some
time. You can remove the directory "temp_folder_concatenated_audio" after this step is complete. It's necessary
for parallel processing of the utterances.

Step 2: Run extract_features.py. This should work without changing anything. Also, this should take a little less
time than the previous step.

Step 3: Run train.py. During the import of utils.py, the files dict.pkl, wordlist.pkl, phones.pkl, and partition.pkl
will be created from dict.txt and wordlist.csv. Towards the top of train.py there is a "trial" string which creates
a folder for that experiment of that name. Also there are options:

TRAIN = True
EVAL = False
LOAD_MODEL = False
PERFORMANCE_EVAL = False
WORDSPLIT = False  # WORDSPLIT and UTTERANCESPLIT (one should be true and other false)
UTTERANCE_SPLIT = True

The example above will train a fresh model using the utterance split partition scheme.

To evaluate the model after training, you need to load one of the checkpoints. In the function "restore_model" of
the Solver class, set G_path = your_model_path. Also you can adjust batch size for your video RAM in
config.train.batch_size.

It should also be noted that config.train.num_epochs is way too large because I always check tensorboard and stop
training when the loss plateaus (around 300k iterations usually).

"""

