import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import get_vocab_size
from pytorch_revgrad import RevGrad
import ipdb

class CTCmodel(nn.Module):
    def __init__(self, config):
        super(CTCmodel, self).__init__()
        self.config = config
        model = eval(self.config.model.name)
        self.model = model(config)

    def forward(self, padded_input, input_lengths):
        """
        Args:
            N is batch_size;
            Ti is the max number of frames;
            D is feature dim;

            padded_input: N x Ti x D
        """
        return self.model(padded_input, input_lengths)

class BLSTM(nn.Module):
    def __init__(self, config):
        super(BLSTM, self).__init__()
        self.config = config
        self.vocab_size = get_vocab_size()
        self.num_mels = config.data.num_mels
        self.hidden_size = config.model.hidden_size
        self.num_layers = config.model.num_layers
        self.batch_first = config.model.batch_first
        self.dropout = config.model.dropout
        self.bidirectional = config.model.bidirectional
        self.lstm = nn.LSTM(input_size=self.num_mels, hidden_size=self.hidden_size,
                             num_layers=self.num_layers, batch_first=self.batch_first,
                             dropout=self.dropout, bidirectional=self.bidirectional)
        self.full1 = nn.Linear(in_features=self.hidden_size if not self.bidirectional else self.hidden_size*2,
                               out_features=500)
        self.adv_layer = nn.Sequential(
                    RevGrad(),
                    nn.Linear(in_features=self.hidden_size if not self.bidirectional else self.hidden_size*2, out_features=2), #21
                    nn.Softmax(dim=-1)
                )
        self.full2 = nn.Linear(in_features=500, out_features=self.vocab_size)

    def forward(self, padded_input, input_lengths):
        lstm_out, _= self.lstm(padded_input)

        x = self.full1(lstm_out)           #CTC branch
        x = F.tanh(x)
        x = self.full2(x)            
        x = F.log_softmax(x, dim=2)  # assign probability
        
        #ipdb.set_trace()

        adv_lstm_in = torch.cat((lstm_out[:, -1, self.hidden_size:], lstm_out[:, 0, self.hidden_size:]), 1)
        b = self.adv_layer(adv_lstm_in)    #adversarial branch
        
        #b = F.log_softmax(b, dim=2)  #get probabilities
        return x, b
