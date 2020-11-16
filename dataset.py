from torch.utils import data
import joblib
from torch.nn.utils.rnn import pad_sequence
import torch
from utils import get_file_metadata, phone2class, class2phone
import yaml
from easydict import EasyDict as edict
import numpy as np
import ipdb
config = edict(yaml.load(open('config.yml'), Loader=yaml.SafeLoader))

class Dataset(data.Dataset):
    'Characterizes a dataset for PyTorch'
    # taken from https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
    def __init__(self, params):
            'Initialization'
            self.list_IDs = params['files']
            self.mode = params["mode"]
            self.wordlist = params['metadata_help']['wordlist']
            self.dictionary = params['metadata_help']['dictionary']
            self.phones = params['metadata_help']['phones']
            self.p2c = phone2class(self.phones)
            self.c2p = class2phone(self.phones)
            self.word_list = joblib.load('word_lists.pkl')

    def __len__(self):
            'Denotes the total number of samples'
            return len(self.list_IDs)

    def __getitem__(self, index):
            'Get the data item'
            file = self.list_IDs[index]
            metadata = get_file_metadata(file, wordlist=self.wordlist, dictionary=self.dictionary)
            """Load data"""
            phones_as_classes = [self.p2c[config.data.SOS_token]]
            [phones_as_classes.append(self.p2c[x]) for x in metadata['phones']]
            while len(phones_as_classes) < 15:
                phones_as_classes.append(self.p2c[config.data.EOS_token])

            if self.mode == 'train':
                spectrogram = joblib.load(file)
                """Get input and target lengths"""
                input_length = spectrogram.shape[0]
                target_length = len(phones_as_classes)
                """Convert to tensors"""
                spectrogram = torch.from_numpy(spectrogram)
                phones_as_classes = torch.FloatTensor(phones_as_classes)
                input_length = torch.FloatTensor([input_length])
                target_length = torch.FloatTensor([target_length])
                
                ww = metadata['word']
                if ww in self.word_list:
                    word = torch.LongTensor([self.word_list.index(ww)])
                else:
                    print("No word in label dictinary!")
                    word = torch.LongTensor([len(self.word_list)])
                #phones_as_classes = pad_sequence(phones_as_classes, batch_first=True, padding_value=self.p2c[config.data.EOS_token])
                #phones_as_classes = self.fix_tensor(phones_as_classes)

                return spectrogram, word, phones_as_classes, input_length, target_length, metadata
            elif self.mode == 'eval':
                """"""
                spectrogram = joblib.load(file)
                """Get input and target lengths"""
                input_length = spectrogram.shape[0]
                target_length = len(phones_as_classes)
                """Convert to tensors"""
                spectrogram = torch.from_numpy(spectrogram)
                phones_as_classes = torch.FloatTensor(phones_as_classes)
                input_length = torch.FloatTensor([input_length])
                target_length = torch.FloatTensor([target_length])
                
                ww = metadata['word']
                if ww in self.word_list:
                    word = torch.LongTensor([self.word_list.index(ww)])
                else:
                    print("No word in label dictinary!")
                    word = torch.LongTensor([len(self.word_list)])

                return spectrogram, word, phones_as_classes, input_length, target_length, metadata

    def fix_tensor(self, x):
        x.requires_grad = True
        x = x.cuda()
        return x
    def collate(self, batch):
        spectrograms = [item[0] for item in batch]
        phones = [item[1] for item in batch]
        input_lengths = [item[2] for item in batch]
        target_lengths = [item[3] for item in batch]
        metadata = [item[4] for item in batch]
        """Extract dysarthric or normal from speaker"""
        # classes = np.identity(2)
        speaker_type = [config.train.dys_class if 'C' not in x['speaker'] else config.train.normal_class for x in metadata]
        speaker_type = np.asarray(speaker_type)
        speaker_type = torch.from_numpy(speaker_type)
        speaker_type = speaker_type.to(dtype=torch.long)
        speaker_type = speaker_type.cuda()
        """"""
        #max_seq = torch.from_numpy(np.zeros((4096, 80)))
        #spectrograms.append(max_seq)
        spectrograms = pad_sequence(spectrograms, batch_first=True, padding_value=0)#[:-1]
        phones = pad_sequence(phones, batch_first=True, padding_value=self.p2c[config.data.EOS_token])
        input_lengths = torch.squeeze(torch.stack(input_lengths))
        target_lengths = torch.squeeze(torch.stack(target_lengths))
        spectrograms = self.fix_tensor(spectrograms)
        phones = self.fix_tensor(phones)
        # input_lengths = self.fix_tensor(input_lengths)
        # target_lengths = self.fix_tensor(target_lengths)

        return {"spectrograms": spectrograms, "phones": phones,
                "input_lengths": input_lengths, "target_lengths": target_lengths,
                "metadata": metadata, "speaker_type": speaker_type}
                
    def collate_transformer(self, batch):
        missed = 0
        spectrograms = [item[0] for item in batch if item[2] < 1601]
        phones = [item[2] for item in batch if item[2] < 1601]
        input_lengths = [item[3] for item in batch if item[2] < 1601]
        target_lengths = [item[4] for item in batch if item[2] < 1601]
        metadata = [item[5] for item in batch if item[2] < 1601]
        missed += (16-len(metadata))
        
        phones_pkl = joblib.load('phones.pkl')
        """Extract dysarthric or normal from speaker"""
        # classes = np.identity(2)
        speaker_type = [config.train.dys_class if 'C' not in x['speaker'] else config.train.normal_class for x in metadata]
        speaker_type = np.asarray(speaker_type)
        speaker_type = torch.from_numpy(speaker_type)
        speaker_type = speaker_type.to(dtype=torch.long)
        speaker_type = speaker_type.cuda()
        """"""
        #max_seq = torch.from_numpy(np.zeros((4096, 80)))
        #spectrograms.append(max_seq)
        spectrograms = pad_sequence(spectrograms, batch_first=True, padding_value=0)#[:-1]
        phones = pad_sequence(phones, batch_first=True, padding_value=self.p2c[config.data.EOS_token])
        input_lengths = torch.squeeze(torch.stack(input_lengths))
        target_lengths = torch.squeeze(torch.stack(target_lengths))
        spectrograms = self.fix_tensor(spectrograms)
        phones = self.fix_tensor(phones)
        # input_lengths = self.fix_tensor(input_lengths)
        # target_lengths = self.fix_tensor(target_lengths)

        return {"spectrograms": spectrograms, "phones": phones,
                "input_lengths": input_lengths, "target_lengths": target_lengths,
                "metadata": metadata, "speaker_type": speaker_type, "skipped_samples":missed}
    
    def collate_maml(batch):
        missed = 0
        spectrograms = [item[0] for item in batch]
        words = [item[1] for item in batch]
        phones = [item[2] for item in batch]
        input_lengths = [item[3] for item in batch]
        target_lengths = [item[4] for item in batch]
        metadata = [item[5] for item in batch]
        missed += (8-len(metadata))

        def fix_tensor(x):
            x.requires_grad = True
            x = x.cuda()
            return x
        phones_pkl = joblib.load('phones.pkl')
        p2c = phone2class(phones_pkl)

        """Extract dysarthric or normal from speaker"""
        # classes = np.identity(2)
        speaker_type = [config.train.dys_class if 'C' not in x['speaker'] else config.train.normal_class for x in metadata]
        speaker_type = np.asarray(speaker_type)
        speaker_type = torch.from_numpy(speaker_type)
        speaker_type = speaker_type.to(dtype=torch.long)
        speaker_type = speaker_type.cuda()
        """"""
        #max_seq = torch.from_numpy(np.zeros((4096, 80)))
        #spectrograms.append(max_seq)
        spectrograms = pad_sequence(spectrograms, batch_first=True, padding_value=0)#[:-1]
        phones = pad_sequence(phones, batch_first=True, padding_value=p2c[config.data.EOS_token])
        input_lengths = torch.squeeze(torch.stack(input_lengths))
        target_lengths = torch.squeeze(torch.stack(target_lengths))
        spectrograms = fix_tensor(spectrograms)
        phones = fix_tensor(phones)
        # input_lengths = self.fix_tensor(input_lengths)
        # target_lengths = self.fix_tensor(target_lengths)

        return {"spectrograms": spectrograms, "words":words, "phones": phones,
                "input_lengths": input_lengths, "target_lengths": target_lengths,
                "metadata": metadata, "speaker_type": speaker_type, "skipped_samples":missed}
    def collate_adv(self, batch):
        """Create dummy tensor of longest length to pad for adversary"""
        dummy_adv_phone_tensor = torch.from_numpy(np.zeros(shape=(15,)))  # 15 is max length of transcription sequence including SOS and EOS
        dummy_adv_phone_tensor = dummy_adv_phone_tensor.to(torch.float32)
        """Most everything else is the same"""
        spectrograms = [item[0] for item in batch]
        num_items = len(spectrograms)
        phones_ = [item[1] for item in batch]
        input_lengths = [item[2] for item in batch]
        target_lengths = [item[3] for item in batch]
        metadata = [item[4] for item in batch]
        """Extract dysarthric or normal from speaker"""
        # classes = np.identity(2)
        speaker_type = [config.train.dys_class if 'C' not in x['speaker'] else config.train.normal_class for x in metadata]
        speaker_type = np.asarray(speaker_type)
        speaker_type = torch.from_numpy(speaker_type)
        speaker_type = speaker_type.to(dtype=torch.long)
        speaker_type = speaker_type.cuda()
        """"""
        spectrograms = pad_sequence(spectrograms, batch_first=True, padding_value=0)
        phones = pad_sequence(phones_, batch_first=True, padding_value=self.p2c[config.data.EOS_token])
        adv_phones = phones_
        adv_phones.append(dummy_adv_phone_tensor)
        adv_phones = pad_sequence(adv_phones, batch_first=True, padding_value=self.p2c[config.data.EOS_token])
        #adv_phones = adv_phones[0:config.train.batch_size, :]
        adv_phones = adv_phones[0: num_items, :]
        # adv_phones_numpy = adv_phones.detach().cpu().numpy()  # for debugging purposes only
        input_lengths = torch.squeeze(torch.stack(input_lengths))
        target_lengths = torch.squeeze(torch.stack(target_lengths))
        spectrograms = self.fix_tensor(spectrograms)
        phones = self.fix_tensor(phones)
        adv_phones = self.fix_tensor(adv_phones)
        # input_lengths = self.fix_tensor(input_lengths)
        # target_lengths = self.fix_tensor(target_lengths)

        return {"spectrograms": spectrograms, "phones": phones,
                "input_lengths": input_lengths, "target_lengths": target_lengths,
                "metadata": metadata, "speaker_type": speaker_type,
                'adv_phones': adv_phones}

    def collate_eval(self, batch):
        spectrograms = [item[0] for item in batch]
        phones = [item[1] for item in batch]
        input_lengths = [item[2] for item in batch]
        target_lengths = [item[3] for item in batch]
        metadata = [item[4] for item in batch]
        spectrograms = pad_sequence(spectrograms, batch_first=True, padding_value=0)
        phones = pad_sequence(phones, batch_first=True, padding_value=self.p2c[config.data.EOS_token])
        input_lengths = torch.squeeze(torch.stack(input_lengths))
        target_lengths = torch.squeeze(torch.stack(target_lengths))
        spectrograms = self.fix_tensor(spectrograms)
        phones = self.fix_tensor(phones)

        return {"spectrograms": spectrograms, "phones": phones,
                "input_lengths": input_lengths, "target_lengths": target_lengths,
                "metadata": metadata}
