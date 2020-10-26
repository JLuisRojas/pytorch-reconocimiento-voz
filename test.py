import math
import pandas as pd
import torch
import torchaudio
import matplotlib.pyplot as plt
from ctcdecode import CTCBeamDecoder
from torch.utils.data import Dataset, DataLoader
from torch import nn
from collections import OrderedDict
import pytorch_lightning as pl
import torch.nn.functional as F

import numpy as np

def wer(r, h):
    """
	https://martin-thoma.com/word-error-rate-calculation/
    Calculation of WER with Levenshtein distance.
    Works only for iterables up to 254 elements (uint8).
    O(nm) time ans space complexity.
    Parameters
    ----------
    r : list
    h : list
    Returns
    -------
    int
    Examples
    --------
    >>> wer("who is there".split(), "is there".split())
    1
    >>> wer("who is there".split(), "".split())
    3
    >>> wer("".split(), "who is there".split())
    3
    """
    # initialisation

    d = np.zeros((len(r) + 1) * (len(h) + 1), dtype=np.uint8)
    d = d.reshape((len(r) + 1, len(h) + 1))
    for i in range(len(r) + 1):
        for j in range(len(h) + 1):
            if i == 0:
                d[0][j] = j
            elif j == 0:
                d[i][0] = i

    # computation
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            if r[i - 1] == h[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                substitution = d[i - 1][j - 1] + 1
                insertion = d[i][j - 1] + 1
                deletion = d[i - 1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)

    return d[len(r)][len(h)]

class VocabEsp:
    def __init__(self):
        self.chars = [
            'a', 'b', 'c',
            'd', 'e', 'f', 'g',
            'h', 'i', 'j',
            'k', 'l', 'm',
            'n', 'ñ', 'o',
            'p', 'q', 'r',
            's', 't', 'u',
            'v', 'w', 'x',
            'y', 'z',
            # Otros caracteres especiales
            ' ']

        self.vocab= {
            '<blank>': 0,
        }

        for i, c in enumerate(self.chars):
            self.vocab[c] = i+1

    def __call__(self, cadena):
        res = []
        cadena = cadena.lower()
        for c in cadena:
            if c in self.vocab:
                res.append(self.vocab[c])

        return res

    def __len__(self):
        return len(self.chars) + 1

class CommonVoiceDataset(Dataset):
    """ Common Voice Dataset """
    def __init__(self,
                 root_dir="./dataset/",
                 audio_distrib='es',
                 distrib='dev',
                 ms=0.01,
                 sample_rate=48000,
                 vocab=None):
        self.root_dir = root_dir
        self.data_dir = f"{root_dir}common-voice/{audio_distrib}/"
        self.clips_dir = f"{self.data_dir}clips/"

        self.audio_distrib = audio_distrib
        self.distrib = distrib
        self.ms = ms
        self.sample_rate = sample_rate

        self.vocab = vocab
        self.vocab_len = len(vocab)

        print(f"Vocab len: {self.vocab_len}")

        self.specgram = torchaudio.transforms.Spectrogram(
            normalized=True,
            n_fft=int(self.ms*self.sample_rate),
            win_length=int(self.ms*self.sample_rate),
            hop_length=int(self.ms*self.sample_rate)
        )

        self.df = pd.read_csv(f"{self.data_dir}{distrib}.tsv", sep='\t')

    def __len__(self):
        return len(self.df.index)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.df.iloc[idx]

        audio_file = row['path']
        sentence = row['sentence']

        audio_path = self.clips_dir + audio_file

        waveform, sample_rate = torchaudio.load(audio_path)

        spec = self.specgram(waveform)
        spec = spec.log2()

        spec = torch.unsqueeze(spec, 0)

        # Quita los valores 
        mask = spec == float('-inf')

        spec[mask] = 0

        mean = torch.mean(spec)
        std = torch.std(spec)

        gain = 1.0 / (torch.max(torch.abs(spec)) + 1e-5)

        spec *= gain

        spec[mask] = -1.0

        # Se corta la parte de alta frequencia
        # solo se deja 125 bins
        spec = spec[:, :125, :] # c, h, w

        # Codifica la oracion
        sentence_encoded = torch.Tensor(self.vocab(sentence))

        sentence_encoded = torch.unsqueeze(sentence_encoded, 0)

        return {
            'features': spec,
            'sentence': sentence_encoded
        }

class SequenceWise(nn.Module):
    def __init__(self, module):
        """
        Colapsa el input de (B, T, F) a (B*T, F) y aplica el modulo
        al input colapsado
        """
        super(SequenceWise, self).__init__()
        self.module = module

    def forward(self, x):
        b, t = x.size(0), x.size(1)
        x = x.view(b * t, -1).contiguous()
        x = self.module(x)
        x = x.view(b, t, -1).contiguous()
        return x

    def __repr__(self):
        tmpstr = self.__class__.__name__ + ' (\n'
        tmpstr += self.module.__repr__()
        tmpstr += ')'
        return tmpstr


class DeepRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(DeepRNN, self).__init__()
        self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size,
                bias=True, bidirectional=True)

    def forward(self, x):
        x, h = self.rnn(x)

        return x

class DeepSpeech2(nn.Module):
    def __init__(self,
                 sample_rate=48000,
                 window_size=0.01,
                 rnn_hidden_size=400,
                 vocab_len=29):
        super(DeepSpeech2, self).__init__()

        self.vocab_len = vocab_len

        self.sample_rate = sample_rate
        self.convs = nn.Sequential(
            # N, 1, 125, L ->
            nn.Conv2d(1, 32, kernel_size=(41, 11), stride=(2, 2), padding=(20,5)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5)),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        # Formula para calcular shape despues de las conv y colapsando los
        # filtros y las features
        # s = (W - K + 2P)/ S+1
        self.rnn_input_size = int(math.floor((sample_rate * window_size) / 2) + 1)
        self.rnn_input_size = int(math.floor(self.rnn_input_size + 2 * 20 - 41) / 2 + 1)
        self.rnn_input_size = int(math.floor(self.rnn_input_size + 2 * 10 - 21) / 2 + 1)
        self.rnn_input_size *= 32

        self.rnn_hidden_size = rnn_hidden_size

        rnns_layers = []

        rnns_layers.append((f"rnn_0", DeepRNN(input_size=self.rnn_input_size,
                hidden_size=self.rnn_hidden_size)))

        for r in range(1, 5):
            rnns_layers.append((f"rnn_{r}",
                DeepRNN(input_size=self.rnn_hidden_size*2,
                hidden_size=self.rnn_hidden_size)))

        self.rnns = nn.Sequential(OrderedDict(rnns_layers))

        # Sequence wise fc, que colapsa las 2 primer dimensiones
        self.fc1 = SequenceWise(nn.Sequential(
            nn.Linear(rnn_hidden_size*2, 1024, bias=False),
            nn.ReLU()
        ))

        self.fc2 = SequenceWise(
            nn.Linear(1024, self.vocab_len, bias=False)
        )

        #print(self.rnn_input_size)

        self.sm = nn.Softmax(dim=-1)


    def forward(self, x):
        _x = self.convs(x)
        # x -> (batch, chanels, features, seq_length)

        #print(_x.shape)

        # Colapsa los filtros y las features
        sizes = _x.size()
        _x = _x.view(sizes[0], sizes[1]*sizes[2], sizes[3]).contiguous()
        # x -> (batch, channels*features, seq_length)
        #print(_x.shape)

        # Permuta el tensor para que sea:
        # (batch, seq_length, features)
        _x = torch.einsum('bfs->bsf', _x)
        #print(_x.shape)

        _x = self.rnns(_x)
        # (batch, seq_length, rnn_hidden_size)
        #print(_x.shape)

        # Aplica la fc a todos los t
        _x = self.fc1(_x)
        _x = self.fc2(_x)

        # Aplica softmax
        _x = self.sm(_x)

        return _x

class PadCollate:
    def __init__(self):
        pass

    def __call__(self, batch):
        batch_size = len(batch)

        features_lengths = []
        sentence_lenghts= []

        for e in batch:
            features_shape = e['features'].shape
            sentence_shape = e['sentence'].shape

            features_lengths.append(features_shape[-1])
            sentence_lenghts.append(sentence_shape[-1])

        max_features_length = max(features_lengths)
        max_sentence_length = max(sentence_lenghts)

        batch_features = []
        batch_sentences = []

        for idx, e in enumerate(batch):
            features = e['features']
            padded_features = F.pad(features, (0, max_features_length - features_lengths[idx]),
                    "constant", 0)

            batch_features.append(padded_features)

            sentence = e['sentence']
            padded_sentence = F.pad(sentence, (0, max_sentence_length -
                sentence_lenghts[idx]), "constant", -1)

            batch_sentences.append(padded_sentence)

        batch_features = torch.cat(batch_features, 0)
        batch_sentences = torch.cat(batch_sentences, 0)

        features_lengths = torch.Tensor(features_lengths).type(torch.int32)
        sentence_lenghts = torch.Tensor(sentence_lenghts).type(torch.int32)

        #print(batch_features.shape)
        #print(batch_sentences.shape)
        #print(features_lengths.shape)
        #print(sentence_lenghts.shape)

        return (batch_features, batch_sentences, features_lengths, sentence_lenghts)

class CVDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=32):
        super().__init__()
        self.batch_size = batch_size

    # TODO: implemetar con las distribuciones correctas
    def setup(self, stage):
        if stage == 'fit':
            self.cv_train = CommonVoiceDataset(vocab=VocabEsp())
            self.cv_val = CommonVoiceDataset(vocab=VocabEsp())

        if stage == 'test':
            self.cv_test = CommonVoiceDataset(vocab=VocabEsp())

    def train_dataloader(self):
        return DataLoader(self.cv_train, num_workers=8, batch_size=self.batch_size, collate_fn=PadCollate())

    def val_dataloader(self):
        return DataLoader(self.cv_val, num_workers=8, batch_size=self.batch_size, collate_fn=PadCollate())

    def test_dataloader(self):
        return DataLoader(self.cv_test, num_workers=8, batch_size=self.batch_size, collate_fn=PadCollate())

class DSModule(pl.LightningModule):
    def __init__(self, params):
        super().__init__()
        self.model = DeepSpeech2()
        self.ctc_loss = nn.CTCLoss(reduction='none')
        self.vocab_str = list('_abcdefghijklmnñopqrstuvwxyz ')
        print(len(self.vocab_str))

        self.ctc_decoder =  CTCBeamDecoder(
            self.vocab_str
        )

    def forward(self, x):
        y = self.model(x)

        return y

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

        return optimizer

    def _ctc_reshape(self, y):
        sizes = y.size()

        fl = torch.full((sizes[0],), sizes[1], dtype=torch.int32)

        # T, N, C
        y = y.view(sizes[1], sizes[0], sizes[2]).contiguous()

        return y, fl

    def _ctc_decode(self, y):
        beam_results, _, _, out_len = self.ctc_decoder.decode(y)

        batch_size, beams, _ = beam_results.size()

        results = []
        for batch_idx in range(batch_size):
            results.append(beam_results[batch_idx][0][:out_len[batch_idx][0]])

        return results

    def _decode(self, sentence):
        sentence = sentence.type(torch.int32)

        string = ''

        for c in sentence:
            string += self.vocab_str[c.item()]

        return string


    def _wer(self, sentences, sentence_lenghts, decoded):
        batch_size, _ = sentences.size()

        sentences_true = []
        sentences_predicted = []

        sum_wer = 0

        for batch_idx in range(batch_size):
            sentence_true = sentences[batch_idx][:sentence_lenghts[batch_idx]]

            sentence_true = self._decode(sentence_true)
            sentence_predicted = self._decode(decoded[batch_idx])

            size_true_words = len(sentence_true.split())

            _wer = wer(sentence_true.split(), sentence_predicted.split())

            _wer = size_true_words / _wer

            if _wer > 1: _wer = 1

            sum_wer += _wer

        return sum_wer / batch_size

    def training_step(self, batch, batch_idx):
        features, sentences, fl, sl = batch

        _y_m = self(features)

        _y, fl = self._ctc_reshape(_y_m)

        loss = self.ctc_loss(_y, sentences, fl, sl).mean()

        decoded = self._ctc_decode(_y_m)

        wer_metric = self._wer(sentences, sl, decoded)

        return -loss

    def validation_step(self, batch, batch_idx):
        features, sentences, fl, sl = batch

        _y = self(features)

        _y, fl = self._ctc_reshape(_y)

        loss = self.ctc_loss(_y, sentences, fl, sl)

        return -loss

    def validation_epoch_end(self, outputs):
        loss = torch.cat([o for o in outputs], 0).mean()

        return {
            'loss': loss
        }

data_module = CVDataModule(batch_size=2)

model = DSModule({})

trainer = pl.Trainer(
    fast_dev_run=True
)

trainer.fit(model, data_module)

"""
dataset_dev = CommonVoiceDataset(vocab=VocabEsp())

loader = DataLoader(dataset_dev, batch_size=2, collate_fn=PadCollate())

model = DeepSpeech2()

features, sentences, fl, sl = next(iter(loader))

y = model(features)

#print(y.shape)

#for batch_ndx, sample in enumerate(loader):
#    features, sentences, fl, sl = sample
#    y = model(features)
#    print(y.shape)


for i in range(16, 17):
    x_i = dataset_dev[i]

    x = x_i['features']
    y = x_i['sentence']

    print(type(x))
    print(x.shape)
    print(y.shape)

    print(x)
    print(torch.max(torch.abs(x)))

    plt.figure()
    plt.imshow(x[0, 0, :, :].numpy(), origin='lower')
    plt.show(block=True)
"""
