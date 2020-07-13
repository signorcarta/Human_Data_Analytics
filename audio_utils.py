import numpy as np
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib.gridspec as gridspec

from CNN import CNN
from python_speech_features import *
from scipy.io import wavfile
from scipy.io.wavfile import read, write


def audio_img(audio_path):
    a = read(audio_path)
    data = np.array(a[1], dtype=float)
    data /= np.max(np.abs(data))

    print("{}".format(str(data)))
    plt.plot(a[1])
    plt.ylabel('some numbers')
    plt.show()


def audio_preprocessed(audio_path):
    numcep = 13
    winstep = 0.01
    prep = CNN.preprocess(audio_path, numcep=numcep, winstep=winstep)
    prep = prep.reshape((len(prep), numcep))
    fig, ax = plt.subplots()
    ax.matshow(prep)
    plt.show()


def audio_spectrogram(audio_path):
    fs, data = wavfile.read(audio_path)
    data = np.array(data, dtype=float)
    data /= np.max(np.abs(data))
    prep = mfcc(data, 16000, winlen=0.025, winstep=0.002, nfilt=1156, nfft=512, lowfreq=0, highfreq=None,
                preemph=0.97, numcep=1156, winfunc=lambda x: np.ones((x,)))

    plt.subplot()
    plt.specgram(data, Fs=16000)  # TODO pick that specgram and pass as input of the network
    plt.show()


def cm_plot(cm, wanted_words):
    df_cm = pd.DataFrame(cm, index=wanted_words, columns=wanted_words)
    plt.figure()
    sn.heatmap(df_cm, annot=True)
    plt.show()


if __name__ == "__main__":
    audio_path = "trainset/speech_commands_v0.02/bed/0b7ee1a0_nohash_0.wav"
    audio_img(audio_path)
    audio_preprocessed(audio_path)
    audio_spectrogram(audio_path)
