import os
import pandas as pd
import pyeeg
import pickle
from tqdm import tqdm
from catboost import CatBoostClassifier
import warnings
from entropy import *
import mne
import pickle as pkl

import logging
logging.getLogger("mne").setLevel(logging.WARNING)

warnings.filterwarnings('ignore')


class EmotionRecognition:
    def __init__(self, path="data_preprocessed_python/", format='dat'):
        def first_der(x):
            return np.mean(np.abs(x[1:] - x[0:-1]))

        def second_der(x):
            return np.mean(np.abs(x[2:] - x[0:-2]))

        def first_der_norm(x):
            return first_der(x / max(np.abs(x)))

        def second_der_norm(x):
            return second_der(x / max(np.abs(x)))

        def samp_entropy(X):
            return sample_entropy(X)

        def spec_entropy(X):
            return spectral_entropy(X, 100)

        def sing_entropy(X):
            return svd_entropy(X)

        def petrosyan(X):
            return petrosian_fd(X)

        def Hig(X):
            return higuchi_fd(X)

        def Katz(X):
            return katz_fd(X)

        def power_4_8(X):
            return pyeeg.bin_power(X, [4, 8], 1000)[0][0]

        def power_8_16(X):
            return pyeeg.bin_power(X, [8, 16], 1000)[0][0]

        def power_16_32(X):
            return pyeeg.bin_power(X, [16, 32], 1000)[0][0]

        def power_32_64(X):
            return pyeeg.bin_power(X, [32, 64], 1000)[0][0]

        def freq_delta(X, Fs=100):
            L = len(X)
            data_fft = np.fft.fft(X)
            frequency = abs(data_fft / L)
            frequency = frequency[: L // 2 + 1] * 2
            delta = frequency[L * 1 // Fs - 1: L * 4 // Fs]
            return np.std(delta)

        def freq_theta(X, Fs=100):
            L = len(X)
            data_fft = np.fft.fft(X)
            frequency = abs(data_fft / L)
            frequency = frequency[: L // 2 + 1] * 2
            theta = frequency[L * 4 // Fs - 1: L * 8 // Fs]
            return np.std(theta)

        def freq_alpha(X, Fs=100):
            L = len(X)
            data_fft = np.fft.fft(X)
            frequency = abs(data_fft / L)
            frequency = frequency[: L // 2 + 1] * 2
            alpha = frequency[L * 5 // Fs - 1: L * 13 // Fs]
            return np.std(alpha)

        def freq_beta(X, Fs=100):
            L = len(X)
            data_fft = np.fft.fft(X)
            frequency = abs(data_fft / L)
            frequency = frequency[: L // 2 + 1] * 2
            beta = frequency[L * 13 // Fs - 1: L * 30 // Fs]
            return np.std(beta)

        def freq_gamma(X, Fs=100):
            L = len(X)
            data_fft = np.fft.fft(X)
            frequency = abs(data_fft / L)
            frequency = frequency[: L // 2 + 1] * 2
            gamma = frequency[L * 30 // Fs - 1: L * 50 // Fs]
            return np.std(gamma)
        self.functions = [np.mean, np.std, first_der,second_der,first_der_norm, second_der_norm,perm_entropy,app_entropy, samp_entropy,
            spec_entropy, sing_entropy, petrosyan, Hig, Katz, power_4_8, power_8_16, power_16_32, power_32_64,
            freq_delta, freq_theta, freq_alpha, freq_beta, freq_gamma]

        def create_df(data, cut=2500, train=True, n_segments=10):
            if format == 'edf' or format == 'bdf':
                n = len(data)
                m = len(data[0])
            elif format == 'dat':
                n = len(data['data'])
                m = len(data['data'][0][0])
            features_all = []
            good_chanels = [2, 3, 4, 5, 8, 12, 14, 32, 30, 26, 22, 21, 20, 18]
            for j in range(n_segments):
                start = cut + j * (m - cut) // n_segments
                finish = cut + (j + 1) * (m - cut) // n_segments
                for i in range(n):
                    features = dict()
                    for ch in good_chanels:
                        for func in self.functions:
                            if format == 'edf' or format == 'bdf':
                                features[f"channel_{ch}_{func.__name__}"] = func(data[i][ch][start:finish])
                            elif format == 'dat':
                                features[f"channel_{ch}_{func.__name__}"] = func(data['data'][i][ch][start:finish])

                    if train and format == 'dat':
                        features['arousal'] = data['labels'][i][0]
                        features['valence'] = data['labels'][i][1]
                        features['dominance'] = data['labels'][i][2]
                        features['liking'] = data['labels'][i][3]

                    features_all.append(features)
            return pd.json_normalize(features_all)

        def create_dataset():
            datasets = []
            for fn in tqdm(os.listdir(path)):
                with open(path + fn, 'rb') as f:
                    if format == 'edf':
                        data = mne.io.read_raw_edf(f).get_data()
                    elif format == 'bdf':
                        data = mne.io.read_raw_bdf(f).get_data()
                    elif format == 'dat':
                        data = pickle.loads(f.read(), encoding='latin1')

                    features = create_df(data)
                    features['man_num'] = fn.split('.')[0]
                    datasets.append(features)

            data_all = pd.concat(datasets)

            data_all = data_all.sample(frac=1)
            return data_all

        '''
        self.dataset = create_dataset()
        self.model_arousal = train_model(get_X(self.dataset), get_Y(self.dataset, 'arousal'))
        self.model_valence = train_model(get_X(self.dataset), get_Y(self.dataset, 'valence'))
        self.model_dominance = train_model(get_X(self.dataset), get_Y(self.dataset, 'dominance'))
        '''

        with open('arousal.pkl', 'rb') as f:
            self.model_arousal = pkl.load(f)
        with open('dominance.pkl', 'rb') as f:
            self.model_dominance = pkl.load(f)
        with open('valence.pkl', 'rb') as f:
            self.model_valence = pkl.load(f)

    def predict_measure(self, data, model):
        return model.predict_proba(data)

    def prepare_raw(self, data, start, finish, n_segments=10, format='edf'):
        n = ''
        if format == 'edf' or format == 'bdf':
            data = data[:, start:finish]
            n = len(data)
        elif format == 'dat':
            n = len(data['data'])
            #TODO
        features_all = []
        good_chanels = [2, 3, 4, 5, 8, 12, 14, 32, 30, 26, 22, 21, 20, 18]

        if format == 'edf' or format == 'bdf':
            features = dict()
            for i in good_chanels:
                for func in self.functions:
                    features[f"channel_{i}_{func.__name__}"] = func(data[i])
            features_all.append(features)

        elif format == 'dat':
            for i in range(n):
                features = dict()
                for ch in good_chanels:
                    for func in self.functions:
                        if format == 'edf' or format == 'bdf':
                            features[f"channel_{ch}_{func.__name__}"] = func(data[ch])
                        elif format == 'dat':
                            features[f"channel_{ch}_{func.__name__}"] = func(data['data'][i][ch])

                features_all.append(features)
        return pd.json_normalize(features_all)

    def start_prediction(self, input, form='edf'):
        def my_max_function(arr, mx):
            ind = 0
            for i in range(len(arr)):
                if arr[i] == mx:
                    ind = i
            return ind

        i = 0
        while i != 1000000:
            data = ''
            if form == 'edf':
                data = mne.io.read_raw_edf(input).get_data()
            elif form == 'bdf':
                data = mne.io.read_raw_bdf(input).get_data()
            elif form == 'dat':
                data = pickle.loads(input.read(), encoding='latin1')['data']

            data_prepared = self.prepare_raw(data, i, i + 15000, format=form)

            arousal = self.predict_measure(data_prepared, self.model_arousal)
            valence = self.predict_measure(data_prepared, self.model_valence)
            dominance = self.predict_measure(data_prepared, self.model_dominance)
            print('Arousal :', max(arousal[0]), 'probability for ', my_max_function(arousal[0], max(arousal[0])))
            print('Valence :', max(valence[0]), 'probability for ', my_max_function(valence[0], max(valence[0])))
            print('Dominance :', max(dominance[0]), 'probability for ', my_max_function(dominance[0], max(dominance[0])))
            print('-------------------------')
            i += 15000


er = EmotionRecognition()
er.start_prediction("s02.bdf", 'bdf')
