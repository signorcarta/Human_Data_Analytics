import numpy as np
import os
import types
import tensorflow as tf
import tensorflow.keras as keras
import dataset_utils

from tensorflow.keras.activations import softmax, relu
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense
from python_speech_features import *
from scipy.io import wavfile
from ASRModel import ASRModel

MODEL_DIR = 'model'
MODEL_NAME = 'model.h5'
RES_DIR = 'res'
JSON_DIR = 'json'


def gen_sample(data, k):
    for v in data:
        yield v[k]


class CNN(ASRModel):

    # constructor
    def __init__(self, model_id: str):
        print("CNN constructor")
        self.model_id = model_id

        # Create model
        model_path = os.path.join(MODEL_DIR, model_id, MODEL_NAME)
        if os.path.isfile(model_path):
            self.model = keras.models.load_model(model_path)
            self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        else:
            # this is a new model
            self.model = Sequential()

        self.wanted_words = ['bed', 'stop', 'seven', 'up']
        # add UNKNOWN label to the dataset if not present
        if dataset_utils.UNKNOWN not in self.wanted_words:
            self.wanted_words.append(dataset_utils.UNKNOWN)

    def preprocess(self, audio):
        """
        from an input path, load the single audio and return preprocessed audio
        which will be the input of the net
        :param audio: input audio path to preprocess
        :return: the preprocessed audio, the model input
        """

        # print("CNN preprocess")
        if isinstance(audio, str) and os.path.isfile(audio):
            fs, data = wavfile.read(audio)
            data = np.array(audio, dtype=float)
            data /= np.max(np.abs(data))
            return mfcc(data, fs, winlen=0.025, winstep=0.01, nfilt=26, nfft=512, lowfreq=0, highfreq=None,
                        preemph=0.97,
                        winfunc=lambda x: np.ones((x,))).reshape((99, 13, 1))
        elif isinstance(audio, np.ndarray):
            data = np.array(audio, dtype=float)
            data /= np.max(np.abs(data))
            data = mfcc(data, 16000, winlen=0.025, winstep=0.01, nfilt=26, nfft=512, lowfreq=0, highfreq=None,
                        preemph=0.97, winfunc=lambda x: np.ones((x,)))
            return data.reshape((99, 13, 1))
        else:
            raise TypeError("Input audio can't be preprocessed, unsupported type: " + str(type(audio)))

    def preprocess_gen(self, audios):
        for data in audios:
            data = np.array(data, dtype=float)
            data /= np.max(np.abs(data))
            yield mfcc(data, 16000, winlen=0.025, winstep=0.01, nfilt=26, nfft=512, lowfreq=0, highfreq=None,
                       preemph=0.97,
                       winfunc=lambda x: np.ones((x,))).reshape((99, 13, 1))

    def batch_preprocessing_gen(self, mnist_val, k_list, ww_size):
        for sample in mnist_val:
            batch = sample[k_list[0]]
            labels = sample[k_list[1]]
            preprocessed_batch = np.array([self.preprocess(data) for data in batch])
            preprocessed_label = np.array(
                [np.concatenate((np.zeros(l), np.array([1.0]), np.zeros(ww_size-l-1))) for l in labels])
            # print(preprocessed_label)
            yield preprocessed_batch, preprocessed_label

    def build_model(self):
        """
        Create the model structure with the parameters specified in the constructor
        :return:
        """
        # add layers [! input shape must be (28,28,1) !]
        self.model.add(Conv2D(64, kernel_size=3, activation=relu, input_shape=(99, 13, 1)))
        self.model.add(Conv2D(32, kernel_size=3, activation=relu))
        self.model.add(Flatten())
        self.model.add(Dense(len(self.wanted_words), activation=softmax))  # 11 nodes at output layer (can be changed)
        # self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.SGD(lr=0.01),
                      metrics=['accuracy'])
        # my_optimizers = [keras.optimizers.Adam(), keras.optimizers.SGD(nesterov=True)]
        # self.model.compile(optimizer=[my_optimizers], loss='categorical_crossentropy', metrics=['accuracy'])

        print("CNN build_model")

    def train(self, trainset):
        """
        Train the builded model in the input dataset specified in the
        :return: the id of the builded model, useful to get the .h5 file
        """
        epochs = 10
        steps_per_epoch = 10
        validation_steps = 3

        t_batch_size = 500
        v_batch_size = 500

        val_percentage = 10.
        test_percentage = 10.
        unknown_percentage = 10.

        if os.path.isdir(trainset):
            x_train, y_train, x_val, y_val, x_test, y_test, info = \
                dataset_utils.load_dataset(trainset, val_percentage=val_percentage, test_percentage=test_percentage)

            g_train = dataset_utils.dataset_generator(x_train, y_train, info, self.wanted_words,
                                                      batch_size=t_batch_size, tot_size=-1,
                                                      unknown_percentage=unknown_percentage, balanced=True)
            g_val = dataset_utils.dataset_generator(x_val, y_val, info, self.wanted_words,
                                                    batch_size=v_batch_size, tot_size=-1,
                                                    unknown_percentage=unknown_percentage)
        else:
            import tensorflow_datasets as tfds

            ds_train, info_train = tfds.load('speech_commands', split=tfds.Split.TRAIN, batch_size=500, with_info=True)
            ds_val, info_val = tfds.load('speech_commands', split=tfds.Split.VALIDATION, batch_size=100, with_info=True)

            g_train = tfds.as_numpy(ds_train)
            g_val = tfds.as_numpy(ds_val)

        # TODO: len(self..wanted_words) is not supported for tfds in batch_preprocessing_gen
        xy_train = self.batch_preprocessing_gen(g_train, ('audio', 'label'), len(self.wanted_words))
        xy_val = self.batch_preprocessing_gen(g_val, ('audio', 'label'), len(self.wanted_words))

        self.model.fit(x=xy_train, epochs=epochs, verbose=2, steps_per_epoch=steps_per_epoch,
                       validation_steps=validation_steps,
                       validation_data=xy_val, use_multiprocessing=False)
        print("CNN train")


    @staticmethod
    def load_model(model_path: str) -> ASRModel:
        """
        load a pretrained model from the specified path
        :param model_path:
        :return:
        """
        if model_path.endswith(os.sep):  # support path ending with '/' or '\'
            model_path = model_path[:-1]
        assert os.path.isdir(model_path), "model_path is not a dir: {}".format(model_path)
        model_id = os.path.basename(model_path)
        print("CNN load_model {}".format(model_path))
        cnn = CNN(model_id)
        # cnn.graph = load_graph()  # "model.h5"
        return cnn

    def save_model(self, path: str):
        """
        Save the current model in the specified path
        :param path:
        :return:
        """

        # self.model.save(path, overwrite=True, include_optimizer=True, save_format=None, signatures=None, options=None)
        print("CNN save_model")

    def test(self, testset_path: str):
        """
        Test the trained model, with
        :return:
        """

        # self.model.predict(testset_path)
        print("CNN test")
