"""
Used for various functions and the creation of:
dict.pkl
wordlist.pkl
phones.pkl
partition.pkl
"""

import os
import pandas as pd
import joblib
from preprocessing import collect_files
import shutil
from tqdm import tqdm
import yaml
from easydict import EasyDict as edict
import random

config = edict(yaml.load(open('config.yml'), Loader=yaml.SafeLoader))

def get_wordlist(wordlist='wordlist.csv'):
    """Read file"""
    wordlist = pd.read_csv(wordlist)
    """Convert to dictionary"""
    wordlist = wordlist.to_dict('r')
    new_wordlist = {}
    for pair in wordlist:
        new_wordlist[pair['FILE NAME']] = pair['WORD']
    wordlist = new_wordlist

    """If the key doesn't contain an underscore, it has B1, B2, and B3
    so we need to make three new keys and delete the old one"""
    keys_to_change = []
    for key, value in wordlist.items():
        if '_' not in key:
            keys_to_change.append(key)
    for key in keys_to_change:
        value = wordlist[key]
        wordlist['B1_' + key] = value
        wordlist['B2_' + key] = value
        wordlist['B3_' + key] = value
        del wordlist[key]

    joblib.dump(wordlist, 'wordlist.pkl')

def get_dictionary(dict='dict.txt'):
    """"""
    new_dict = {}
    with open(dict) as f:
        for l in f:
            l = l.replace('\n', '')
            word = l.split(' ')[0]
            phones = l.split(' ')[1:]
            new_dict[word] = phones
    joblib.dump(new_dict, 'dict.pkl')

def get_phones():
    dictionary = joblib.load('dict.pkl')
    phones = []
    for key, value in dictionary.items():
        phones.extend(value)
    phones = list(set(phones))

    """Add in the PAD, SOS, and EOS tokens"""
    phones.append(config.data.PAD_token)
    phones.append(config.data.SOS_token)
    phones.append(config.data.EOS_token)

    """Write to disk"""
    joblib.dump(phones, 'phones.pkl')

def get_partition_wordsplit():
    wordlist = joblib.load(('wordlist.pkl'))
    unique_words = []
    [unique_words.append(value) for key, value in wordlist.items()]
    unique_words = list(set(unique_words))
    random.shuffle(unique_words)
    split_index = int(config.data.train_test_frac*len(unique_words))
    train = unique_words[0:split_index]
    test = unique_words[split_index:]
    # overlap = (set(train)).intersection(set(test))
    """Dump to disk"""
    joblib.dump({'train': train, 'test': test}, 'partition.pkl')

def get_file_metadata(file, wordlist=None, dictionary=None):
    """Return speaker, word, and phones of given file
    If wordlist or dictionary aren't provided, load them from file"""
    if wordlist == None:
        wordlist = joblib.load('wordlist.pkl')
    if dictionary == None:
        dictionary = joblib.load('dict.pkl')
    utterance = file.split('/')[-1]
    speaker = utterance.split('_')[0]
    delimiter = '_'
    word = wordlist[delimiter.join((utterance.split('_')[1], utterance.split('_')[2]))]
    phones = dictionary[word.upper()]
    return {'speaker': speaker, 'word': word, 'phones': phones, 'utterance': utterance[:-4]}

def get_partition_utterance_split():
    """Set aside one utterance for each speaker of each word for test set and val set"""
    wordlist = joblib.load('wordlist.pkl')
    dictionary = joblib.load('dict.pkl')
    speaker_test_files = {}
    speaker_val_files = {}
    test_files = []
    val_files = []
    all_files = collect_files(config.directories.features)
    for file in all_files:
        metadata = get_file_metadata(file, wordlist=wordlist, dictionary=dictionary)
        if metadata["speaker"] not in speaker_test_files:
            speaker_test_files[metadata["speaker"]] = {}
        if metadata["speaker"] not in speaker_val_files:
            speaker_val_files[metadata["speaker"]] = {}
        if metadata["word"] not in speaker_test_files[metadata["speaker"]]:
            speaker_test_files[metadata["speaker"]][metadata["word"]] = file
            test_files.append(file)
        if metadata["word"] in speaker_test_files[metadata["speaker"]] \
                and metadata["word"] not in speaker_val_files[metadata["speaker"]] \
                and speaker_test_files[metadata["speaker"]][metadata["word"]] != file:
            speaker_val_files[metadata["speaker"]][metadata["word"]] = file
            val_files.append(file)
    """All files stored in test_files is the test set
       All files stored in val_files is val set
       The list of all other files is the train set"""

    """Let's check that there's no intersection between test and val sets"""
    # test_val_intersection = list(set(test_files).intersection(set(val_files)))
    train_files = list(set(all_files) - set(test_files) - set(val_files))
    # intersection_test = (set(train_files)).intersection(set(test_files))  # check there are no overlapping elements
    # intersection_val = (set(train_files)).intersection(set(val_files))
    joblib.dump({'train': train_files, 'test': test_files, 'val': val_files}, 'partition.pkl')


if not os.path.exists('wordlist.pkl'):
    get_wordlist('wordlist.csv')

if not os.path.exists('dict.pkl'):
    get_dictionary('dict.txt')

if not os.path.exists('phones.pkl'):
    get_phones()

if not os.path.exists('partition.pkl'):
    get_partition_utterance_split()


def phone2class(phones=None):
    if phones == None:
        phones = joblib.load('phones.pkl')
    p2c = {}
    # p2c = {config.data.PAD_token: 0, config.data.SOS_token: 1, config.data.EOS_token:2}
    for i, phone in enumerate(phones):
        p2c[phone] = i + 3
    p2c[config.data.PAD_token] = 0
    p2c[config.data.SOS_token] = 1
    p2c[config.data.EOS_token] = 2
    return p2c

def class2phone(phones=None):
    p2c = phone2class(phones)
    c2p = {}
    for key, value in p2c.items():
        c2p[value] = key
    return c2p

def dummy_check_metadata():
    """I used this to check that my assumptions about the naming convention
    are correct (B1, B2, B3 need to be added to certain words)
    After listening, the audios labeled with their words are actually those words so assumptions are correct"""
    files = collect_files('silence_removed')
    dummy_dir = 'dummy_files_check'
    if not os.path.isdir(dummy_dir):
        os.mkdir(dummy_dir)
    for i, file in tqdm(enumerate(files)):
        metadata = get_file_metadata(file)
        target_path = os.path.join(dummy_dir,
                                   metadata['speaker'] + '_' + metadata['word'] + '_' + str(i) + '.wav')
        shutil.copy(file, target_path)

def get_vocab_size():
    dictionary = joblib.load('phones.pkl')
    return len(dictionary)


def main():
    """"""
    # p2c = phone2class()
    # c2p = class2phone()
    # vocab_size = get_vocab_size()


if __name__ == '__main__':
    main()