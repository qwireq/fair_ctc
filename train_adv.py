import os
from tqdm import tqdm
import numpy as np
import joblib
import torch
import torch.nn as nn
import model_adv
import yaml
from easydict import EasyDict as edict
import shutil
from preprocessing import collect_files
import utils
from dataset import Dataset
from torch.utils import data
from itertools import groupby
import json

import ipdb
from sklearn.preprocessing import LabelEncoder
os.environ['CUDA_VISIBLE_DEVICES'] = "4"

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='uint8')[y]


config = edict(yaml.load(open('config.yml'), Loader=yaml.SafeLoader))

if not os.path.exists(config.directories.exps):
    os.mkdir(config.directories.exps)

trial = 'bin_adv_batch32_2nd_trial'#('Trial_unfair_adv_AIM_lab') - without revgrad, just train discriminator
model_name = 'bin_adv_batch32_2nd_trial'
exp_dir = os.path.join(config.directories.exps, trial)
if not os.path.isdir(exp_dir):
    os.mkdir(exp_dir)

TRAIN = False
EVAL = True
LOAD_MODEL = True
PERFORMANCE_EVAL = True
WORDSPLIT = False  # WORDSPLIT and UTTERANCESPLIT (one should be true and other false)
UTTERANCE_SPLIT = True
LAMBDA_ADV = 100

class Solver(object):
    """Solver"""

    def __init__(self):
        """Initialize configurations."""
        
        # Training configurations.
        self.g_lr = config.model.lr
        self.torch_type = torch.float32
        self.lambda_adv = LAMBDA_ADV
        # Miscellaneous.
        self.use_tensorboard = True
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:{}'.format(0) if self.use_cuda else 'cpu')

        # Directories.
        self.log_dir = os.path.join(exp_dir, 'logs')
        self.model_save_dir = os.path.join(exp_dir, 'models')
        self.train_data_dir = config.directories.features
        self.predict_dir = os.path.join(exp_dir, 'predictions')

        if not os.path.isdir(self.log_dir):
            os.mkdir(self.log_dir)
        if not os.path.isdir(self.model_save_dir):
            os.mkdir(self.model_save_dir)
        if not os.path.isdir(self.predict_dir):
            os.mkdir(self.predict_dir)

        """Partition file"""
        if TRAIN:  # only copy these when running a training session, not eval session
            # copy partition to exp_dir then use that for trial (just in case you change partition for other trials)
            shutil.copy(src='partition.pkl', dst=os.path.join(exp_dir, 'partition.pkl'))
            self.partition = os.path.join(exp_dir, 'partition.pkl')
            # copy config as well
            shutil.copy(src='config.yml', dst=os.path.join(exp_dir, 'config.yml'))
            # copy dict
            shutil.copy(src='dict.pkl', dst=os.path.join(exp_dir, 'dict.pkl'))
            # copy phones
            shutil.copy(src='phones.pkl', dst=os.path.join(exp_dir, 'phones.pkl'))
            # copy wordlist
            shutil.copy(src='wordlist.pkl', dst=os.path.join(exp_dir, 'wordlist.pkl'))
            shutil.copy(src='model_adv.py', dst=os.path.join(exp_dir, 'model_adv.py'))

        # Step size.
        self.log_step = config.train.log_step
        self.model_save_step = config.train.model_save_step

        # Build the model
        self.build_model()
        if EVAL or LOAD_MODEL:
            self.restore_model()
        if self.use_tensorboard:
            self.build_tensorboard()

    def build_model(self):
        """Build the model"""
        self.G = model_adv.CTCmodel(config)
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr)
        self.print_network(self.G, 'G')
        self.G.to(self.device)

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

    def print_optimizer(self, opt, name):
        print(opt)
        print(name)

    def build_tensorboard(self):
        """Build a tensorboard logger."""
        from logger import Logger
        self.logger = Logger(self.log_dir)

    def _load(self, checkpoint_path):
        if self.use_cuda:
            checkpoint = torch.load(checkpoint_path)
        else:
            checkpoint = torch.load(checkpoint_path,
                                    map_location=lambda storage, loc: storage)
        return checkpoint

    def restore_model(self):
        """Restore the model"""
        print('Loading the trained models... ')
        # G_path = './exps/TRIAL_6_fair/models/275000-G.ckpt'
        # G_path = './exps/TRIAL_7_mid_high_dys_spks_only/models/400000-G.ckpt'
        # G_path = './exps/TRIAL_4_mid_high_dys_spks_only_lambda_0.5/models/360000-G.ckpt'
        # G_path = './exps/TRIAL_5_mid_high_dys_spks_only_lambda_1/models/295000-G.ckpt'
        # G_path = './exps/TRIAL_6_mid_high_dys_spks_only_lambda_0.5/models/175000-G.ckpt'
        G_path = './exps/'+trial+'/models/70000-G.ckpt' #
        #G_path = './exps/bin_adv/models/30000-G.ckpt' #
        g_checkpoint = self._load(G_path)
        self.G.load_state_dict(g_checkpoint['model'])
        self.g_optimizer.load_state_dict(g_checkpoint['optimizer'])
        self.g_lr = self.g_optimizer.param_groups[0]['lr']

    def update_lr(self, g_lr):
        """Decay learning rates of g"""
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = g_lr

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()

    def wordlist_to_dict(self, words):
        train = {}
        test = {}
        for value in words['train']:
            train[value] = ''
        for value in words['test']:
            test[value] = ''
        return {'train': train, 'test': test}

    def filter_speakers(self, files):
        """"""
        new_files = []
        speakers = []
        for file in tqdm(files):
            speaker = (file.split('/')[-1]).split('_')[0]
            if speaker not in config.data.ignore_speakers:
                new_files.append(file)
                speakers.append(speaker)

        # speakers = set(speakers)  # check that it worked properly
        return new_files

    def get_train_test_wordsplit(self):
        wordlist = joblib.load('wordlist.pkl')
        dictionary = joblib.load('dict.pkl')
        """Get the triain and test files"""
        if not os.path.exists(os.path.join(exp_dir, 'train_test_files.pkl')):
            words = joblib.load(self.partition)
            """Turn train and test into dicts for fast check (hashable)"""
            words = self.wordlist_to_dict(words)
            train_files = []
            test_files = []
            for file in tqdm(collect_files(self.train_data_dir)):
                metadata = utils.get_file_metadata(file, wordlist=wordlist, dictionary=dictionary)
                try:
                    dummy = words['train'][metadata['word']]
                    train_files.append(file)
                except:
                    try:
                        dummy = words['test'][metadata['word']]
                        test_files.append(file)
                    except:
                        print("File in neither train nor test set...")
            joblib.dump({'train': train_files, 'test': test_files}, os.path.join(exp_dir, 'train_test_files.pkl'))
        else:
            files = joblib.load(os.path.join(exp_dir, 'train_test_files.pkl'))
            train_files = files['train']
            test_files = files['test']
        return self.filter_speakers(train_files), self.filter_speakers(test_files)

    def get_train_test_utterance_split(self):
        partition = joblib.load("partition.pkl")
        train_files = self.filter_speakers(partition["train"])
        test_files = self.filter_speakers(partition["test"])
        val_files = self.filter_speakers(partition["val"])
        return train_files, test_files, val_files

    def val_loss(self, val, iterations):
        """Time to write this function"""
        self.val_history = {}
        ######
        classes = ['CF02', 'CF03', 'CF04', 'CF05', 'CM01', 'CM04', 'CM05', 'CM06', 'CM08', 'CM09', 'CM10', 'CM12', 'CM13', 'F04', 'F05', 'M05', 'M08', 'M09', 'M10', 'M11', 'M14']
        classes_bin = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
        ######
        correct = 0
        incorrect = 0
        for batch_number, features in tqdm(enumerate(val)):
            spectrograms = features['spectrograms']
            phones = features['phones']
            input_lengths = features['input_lengths']
            target_lengths = features['target_lengths']
            metadata = features["metadata"]
            batch_speakers = [x['speaker'] for x in metadata]
            self.G = self.G.eval()
            ######
            labels = [classes_bin[classes.index(x['speaker'])] for x in metadata]
            #labels = [classes.index(x['speaker']) for x in metadata]
            labels = np.array(labels)
            labels = torch.from_numpy(labels).long().to(self.device)
            ######
            """Make input_lengths and target_lengths torch ints"""
            input_lengths = input_lengths.to(torch.int32)
            target_lengths = target_lengths.to(torch.int32)
            phones = phones.to(torch.int32)
            ######
            outputs_ctc, outputs_adv  = self.G(spectrograms, input_lengths.long())

            outputs_ctc = outputs_ctc.permute(1, 0, 2)  # swap batch and sequence length dimension for CTC loss

            loss = self.ctc_loss(log_probs=outputs_ctc, targets=phones,
                                 input_lengths=input_lengths, target_lengths=target_lengths)
            loss_adv = self.loss_adv(outputs_adv, labels)
            total_loss = loss + loss_adv*self.lambda_adv

            
            _, predicted = torch.max(outputs_adv.data, -1)
            gt = labels.data
            correct += (predicted == gt).float().sum().cpu().data
            incorrect += (predicted != gt).float().sum().cpu().data
            
            ######
            #ipdb.set_trace()
            """Update the loss history MUST BE SEPARATE FROM TRAINING"""
            self.update_history_val(total_loss, batch_speakers)
        """We have the history, now do something with it"""
        val_loss_means = {}
        for key, value in self.val_history.items():
            val_loss_means[key] = np.mean(np.asarray(value))
        val_loss_means_sorted = {k: v for k, v in sorted(val_loss_means.items(), key=lambda item: item[1])}
        weights = {}
        counter = 1
        val_loss_value = 0
        for key, value in val_loss_means_sorted.items():
            val_loss_value += (config.train.fairness_lambda * counter + (1-config.train.fairness_lambda) * 1) * value
            counter += 1
        val_accuracy = (100 * correct / (correct + incorrect))
        return val_loss_value, val_accuracy

    def update_history(self, loss, speakers):
        """Update the history with the new loss values"""
        loss_copy = loss.detach().cpu().numpy()
        for loss_value, speaker in zip(loss_copy, speakers):
            speaker_index = self.s2i[speaker]
            """Extract row corresponding to speaker"""
            history_row = self.history[speaker_index]
            """Shift all elements by 1 to the right"""
            history_row = np.roll(history_row, shift=1)
            """Overwrite the first value (the last value in the array rolled to the front and is overwritten"""
            history_row[0] = loss_value
            """Set the history row equal to the modified row"""
            self.history[speaker_index] = history_row

    def update_history_val(self, loss, speakers):
        """Update the val_history with the new loss values"""
        loss_copy = loss.detach().cpu().numpy()
        for loss_value, speaker in zip(loss_copy, speakers):
            speaker_index = self.s2i[speaker]
            if speaker_index not in self.val_history:
                self.val_history[speaker_index] = []
            self.val_history[speaker_index].append(loss_value)

    def get_loss_weights(self, speakers, type='fair'):
        """Use self.history to determine the ranking of which category is worst"""
        mean_losses = np.mean(self.history, axis=1)
        """Sort lowest to highest"""
        order_indices = np.argsort(mean_losses)
        """Create weights as in Dr. Hasegawa-Johnson's slides (weight is number of classes performing better)
           We add one to each so that every class has some weight in the loss"""
        weights = np.linspace(1, mean_losses.shape[0], mean_losses.shape[0])
        """Assign the weights according to the proper order"""
        class_weights = {}
        for index, i in enumerate(order_indices):
            class_weights[i] = weights[index]
        """Now grab the correct weight for each speaker"""
        loss_weights = []
        for speaker in speakers:
            loss_weights.append(class_weights[self.s2i[speaker]])
        if type == 'fair':
            """Add in the lambda weighting for fair and unfair training"""
            unfair_weights = np.ones(shape=(len(loss_weights, )))
            loss_weights = np.asarray(loss_weights)

            """Lambda part"""
            loss_weights = config.train.fairness_lambda * loss_weights + (1-config.train.fairness_lambda) * unfair_weights

        elif type == 'unfair':
            """All class losses are weighted evenly, unfair"""
            loss_weights = np.ones(shape=(len(loss_weights,)))

        loss_weights = torch.from_numpy(loss_weights)
        loss_weights = self.fix_tensor(loss_weights)
        return loss_weights

    def speaker2index_and_index2speaker(self):
        self.s2i = {}
        self.i2s = {}
        for i, speaker in enumerate(list(set(config.data.speakers) - set(config.data.ignore_speakers))):
            self.s2i[speaker] = i
            self.i2s[i] = speaker

    def train(self):
        """Create speaker2index and index2speaker"""
        self.speaker2index_and_index2speaker()
        """Initialize history matrix"""
        self.history = np.random.normal(loc=0, scale=0.1, size=(len(self.s2i), config.train.class_history))
        """"""
        """"""
        iterations = 0
        """Get train/test"""
        if WORDSPLIT:
            train, test = self.get_train_test_wordsplit()
        elif UTTERANCE_SPLIT:
            train, test, val = self.get_train_test_utterance_split()
        wordlist = joblib.load('wordlist.pkl')
        dictionary = joblib.load('dict.pkl')
        phones = joblib.load('phones.pkl')
        metadata_help = {'wordlist': wordlist, 'dictionary': dictionary, 'phones': phones}
        p2c = utils.phone2class(phones)
        c2p = utils.class2phone(phones)
        """CTC loss"""
        # self.ctc_loss = nn.CTCLoss(blank=p2c[config.data.PAD_token], reduction='mean')
        self.ctc_loss = nn.CTCLoss(blank=p2c[config.data.PAD_token], reduction='none')
        self.loss_adv = nn.CrossEntropyLoss(reduction = 'none')
        for epoch in range(config.train.num_epochs):
            """Make dataloader"""
            train_data = Dataset({'files': train, 'mode': 'train', 'metadata_help': metadata_help})
            train_gen = data.DataLoader(train_data, batch_size=config.train.batch_size,
                                        shuffle=True, collate_fn=train_data.collate, drop_last=True)
            val_data = Dataset({'files': val, 'mode': 'train', 'metadata_help': metadata_help})
            val_gen = data.DataLoader(val_data, batch_size=config.train.batch_size,
                                        shuffle=True, collate_fn=val_data.collate, drop_last=True)
            ####################
            '''
            y_train = []
            for features in train_gen:
                metadata = features["metadata"]
                for x in metadata:
                    y_train.append(x['speaker'])
            y_train = np.array(y_train)
            lb = LabelEncoder()
            y_train = to_categorical(lb.fit_transform(y_train.ravel()), 21)
            print("save classes...")
            np.save('classes'+str(epoch)+'.npy', lb.classes_)
            
            y_train = np.argmax(y_train, axis=1)
            y_train = torch.from_numpy(y_train).long()
            '''
            classes = ['CF02', 'CF03', 'CF04', 'CF05', 'CM01', 'CM04', 'CM05', 'CM06', 'CM08', 'CM09', 'CM10', 'CM12', 'CM13', 'F04', 'F05', 'M05', 'M08', 'M09', 'M10', 'M11', 'M14']
            classes_bin = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
            ###################
            
                                
            correct = 0
            incorrect = 0
            for batch_number, features in enumerate(train_gen):
                spectrograms = features['spectrograms']
                phones = features['phones']
                input_lengths = features['input_lengths']
                target_lengths = features['target_lengths']
                metadata = features["metadata"]
                batch_speakers = [x['speaker'] for x in metadata]
                ###########
                #labels = np.zeros((config.train.batch_size, 21))
                #for it, x in enumerate(metadata):
                    #labels[it, classes.index(x['speaker'])] = 1 
                labels = [classes_bin[classes.index(x['speaker'])] for x in metadata]
                #labels = [classes.index(x['speaker']) for x in metadata]
                labels = np.array(labels)
                labels = torch.from_numpy(labels).long().to(self.device)
            
                self.G = self.G.train()

                """Make input_lengths and target_lengths torch ints"""
                input_lengths = input_lengths.to(torch.int32)
                target_lengths = target_lengths.to(torch.int32)
                phones = phones.to(torch.int32)
                #########
                #ipdb.set_trace()
                outputs_ctc, outputs_adv = self.G(spectrograms, input_lengths.long())
                outputs_ctc = outputs_ctc.permute(1, 0, 2)  # swap batch and sequence length dimension for CTC loss

                loss = self.ctc_loss(log_probs=outputs_ctc, targets=phones,
                                     input_lengths=input_lengths, target_lengths=target_lengths)
                loss_adv = self.loss_adv(outputs_adv, labels)
                #ipdb.set_trace()

                _, predicted = torch.max(outputs_adv.data, -1)
                gt = labels.data
                correct += (predicted == gt).float().sum().cpu().data
                incorrect += (predicted != gt).float().sum().cpu().data
                #ipdb.set_trace()

                #ipdb.set_trace()

                total_loss = loss + loss_adv*self.lambda_adv
                #########
                """Update the loss history"""
                self.update_history(total_loss, batch_speakers)
                if epoch >= config.train.regular_epochs:
                    loss_weights = self.get_loss_weights(batch_speakers, type='fair')
                else:
                    loss_weights = self.get_loss_weights(batch_speakers, type='unfair')
                total_loss = total_loss * loss_weights
                #ipdb.set_trace()
                # Backward and optimize.
                self.reset_grad()
                # loss.backward()
                total_loss.sum().backward()
                self.g_optimizer.step()
                #counter += 1
                if iterations % self.log_step == 0:
                    accuracy = (100 * correct / (correct + incorrect)).item()
                    print(str(iterations) + ', losses (total, ctc, adv): ' + str(total_loss.sum().item()) + ',' + str(loss.sum().item()) + ',' + str(loss_adv.sum().item()) , 'adv. accuracy: '+str(accuracy))
                    correct = 0
                    incorrect = 0
                    if self.use_tensorboard:
                        self.logger.scalar_summary('loss', total_loss.sum().item(), iterations)
                        self.logger.scalar_summary('accuracy', accuracy, iterations)

                if iterations % self.model_save_step == 0:
                    if self.lambda_adv > 6:
                        self.lambda_adv = self.lambda_adv // 2
                    elif self.lambda_adv == 6:
                        self.lambda_adv -= 1
                    """Calculate validation loss"""
                    val_loss, val_accuracy = self.val_loss(val=val_gen, iterations=iterations)
                    print(str(iterations) + ', val_loss: ' + str(val_loss), 'val_adv. accuracy: '+str(val_accuracy.item()))
                    if self.use_tensorboard:
                        self.logger.scalar_summary('val_loss', val_loss, iterations)
                        self.logger.scalar_summary('val_accuracy', val_accuracy.item(), iterations)
                """Save model checkpoints."""
                if iterations % self.model_save_step == 0:
                    G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(iterations))
                    torch.save({'model': self.G.state_dict(),
                                'optimizer': self.g_optimizer.state_dict()}, G_path)
                    print('Saved model checkpoints into {}...'.format(self.model_save_dir))

                iterations += 1

    def eval(self):

        """Evaluate trained model on test set"""
        if WORDSPLIT:
            train, test = self.get_train_test_wordsplit()
        elif UTTERANCE_SPLIT:
            train, test, val = self.get_train_test_utterance_split()
        wordlist = joblib.load('wordlist.pkl')
        dictionary = joblib.load('dict.pkl')
        phones = joblib.load('phones.pkl')
        metadata_help = {'wordlist': wordlist, 'dictionary': dictionary, 'phones': phones}
        p2c = utils.phone2class(phones)
        c2p = utils.class2phone(phones)
        """Get test generator"""
        test_data = Dataset({'files': test, 'mode': 'eval', 'metadata_help': metadata_help})
        test_gen = data.DataLoader(test_data, batch_size=1,
                                    shuffle=True, collate_fn=test_data.collate_eval, drop_last=True)
        for batch_number, features in tqdm(enumerate(test_gen)):
            spectrograms = features['spectrograms']
            phones = features['phones']
            batch_metadata = features['metadata'][0]
            input_lengths = features['input_lengths']
            self.G = self.G.eval()

            outputs, _ = self.G(spectrograms, input_lengths.unsqueeze(0).long())
            outputs = np.squeeze(outputs.detach().cpu().numpy())
            phones = np.squeeze(phones.detach().cpu().numpy())
            phones = phones.astype(dtype=int)
            phones = [c2p[x] for x in phones]

            output_classes = np.argmax(outputs, axis=1)

            """Decode the output predictions into a phone sequence"""
            # https://stackoverflow.com/questions/38065898/how-to-remove-the-adjacent-duplicate-value-in-a-numpy-array
            duplicates_eliminated = np.asarray([k for k, g in groupby(output_classes)])
            blanks_eliminated = duplicates_eliminated[duplicates_eliminated != 0]
            predicted_phones_ = [c2p[x] for x in blanks_eliminated]
            """remove SOS and EOS"""
            predicted_phones = []
            for x in predicted_phones_:
                if x != 'SOS' and x != 'EOS':
                    predicted_phones.append(x)

            data_to_save = {'speaker': batch_metadata['speaker'],
                            'word': batch_metadata['word'],
                            'true_phones': batch_metadata['phones'],
                            'predicted_phones': predicted_phones}
            dump_path = os.path.join(self.predict_dir, batch_metadata['utterance'] + '.pkl')
            joblib.dump(data_to_save, dump_path)

    def performance(self):
        """"""
        speakers = {}
        for file in tqdm(collect_files(self.predict_dir)):
            utterance = file.split('/')[-1]
            speaker = utterance.split('_')[0]
            if speaker not in speakers:
                speakers[speaker] = [joblib.load(file)]
            else:
                speakers[speaker].append(joblib.load(file))

        delimiter = '_'
        """Percent correct words"""
        correct_words = {}
        WER = {}
        for speaker, utt_list in speakers.items():
            correct_word_count = 0
            for i, utt in enumerate(utt_list):
                true_seq = delimiter.join(utt['true_phones'])
                pred_seq = delimiter.join(utt['predicted_phones'])
                if true_seq == pred_seq:
                    correct_word_count += 1
            word_accuracy = correct_word_count/(i+1)
            correct_words[speaker] = word_accuracy
            WER[speaker] = (1 - word_accuracy) * 100

        """Percent phones recognized in the utterance that are in the true phones"""
        delimiter = '_'
        correct_perc = {}
        for speaker, utt_list in speakers.items():
            correct_word_perc = 0
            for i, utt in enumerate(utt_list):
                true_phones = set(utt['true_phones'])
                pred_phones = set(utt['predicted_phones'])
                intersection = true_phones.intersection(pred_phones)
                percent_correct_phones = len(list(intersection))/len(list(true_phones))
                correct_word_perc += percent_correct_phones
            word_accuracy = correct_word_perc / (i + 1)
            correct_perc[speaker] = word_accuracy

        self.dump_json(dict=correct_words, path=os.path.join(exp_dir, 'test_accuracies_'+model_name+'.json'))
        """Let's sort best to worst WER"""
        sorted_WER = {k: v for k, v in sorted(WER.items(), key=lambda item: item[1])}
        self.dump_json(dict=sorted_WER, path=os.path.join(exp_dir, 'test_WER_'+model_name+'.json'))

        stats = {}
        """Add more code for different stats on the results here"""
        """Let's compare dysarthric performance to normal performance"""
        stats['dysarthric_mean_WER'] = np.mean(np.asarray([value for key, value in WER.items() if 'C' not in key]))
        stats['normal_mean_WER'] = np.mean(np.asarray([value for key, value in WER.items() if 'C' in key]))
        stats['female_WER'] = np.mean(np.asarray([value for key, value in WER.items() if 'F' in key]))
        stats['male_WER'] = np.mean(np.asarray([value for key, value in WER.items() if 'M' in key]))
        self.dump_json(dict=stats, path=os.path.join(exp_dir, 'test_stats_'+model_name+'.json'))


    def to_gpu(self, tensor):
        tensor = tensor.to(self.torch_type)
        tensor = tensor.to(self.device)
        return tensor

    def fix_tensor(self, x):
        x.requires_grad = True
        x = x.to(self.torch_type)
        x = x.cuda()
        return x

    def dump_json(self, dict, path):
        a_file = open(path, "w")
        json.dump(dict, a_file, indent=2)
        a_file.close()

def main():
    solver = Solver()
    if TRAIN:
        solver.train()
    if EVAL:
        solver.eval()
    if PERFORMANCE_EVAL:
        solver.performance()


if __name__ == "__main__":
    main()