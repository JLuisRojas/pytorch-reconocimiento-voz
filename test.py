import pandas as pd
import torch
import torchaudio
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch import nn
import pytorch_lightning as pl
import torch.nn.functional as F

class VocabEsp:
    def __init__(self):
        self.chars = [
            'a', 'b', 'c',
            'd', 'f', 'g',
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
        return len(self.chars)

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

        # Quita los valores 
        #spec[spec == float('-inf')] = 0

        # Se corta la parte de alta frequencia
        # solo se deja 125 bins
        #spec = spec[:, :125, :] # c, h, w

        # Codifica la oracion
        sentence_encoded = torch.Tensor(self.vocab(sentence))

        sentence_encoded = torch.unsqueeze(sentence_encoded, 0)

        return {
            'features': spec,
            'sentence': sentence_encoded
        }

class DeepSpeech2(nn.Module):
    def __init__(self,
                 sample_rate=48000):
        super(DeepSpeech2, self).__init__()

        self.sample_rate = sample_rate
        self.convs = nn.Sequential(
            # N, 1, 125, L ->
            nn.Conv2d(1, 32, kernel_size=(41, 11), stride=(2, 2), padding=(20,5)),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5)),
            nn.ReLU(),
        )

        # Formula para calcular shape
        # s = (W - K + 2P)/ S+1

    def forward(self, x):
        _x = self.convs(x)
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

            features_lengths.append(features_shape[2])
            sentence_lenghts.append(sentence_shape[1])

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

        features_lengths = torch.Tensor(features_lengths)
        sentence_lenghts = torch.Tensor(sentence_lenghts)

        print(batch_features.shape)
        print(batch_sentences.shape)
        print(features_lengths.shape)
        print(sentence_lenghts.shape)

        return (batch_features, batch_sentences, features_lengths, sentence_lenghts)

dataset_dev = CommonVoiceDataset(vocab=VocabEsp())

loader = DataLoader(dataset_dev, batch_size=2, collate_fn=PadCollate())

for batch_ndx, sample in enumerate(loader):
    print(batch_ndx)
    # print(sample)

"""
for i in range(16, 17):
    x_i = dataset_dev[i]

    x = x_i['features']
    y = x_i['sentence']

    print(type(x))
    print(x.shape)
    print(y.shape)

    print(x)

    plt.figure()
    plt.imshow(x[0, :, :].numpy(), origin='lower')
    plt.show(block=True)
"""
