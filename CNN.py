import numpy as np
import os
import tensorflow.keras as keras
import dataset_utils
import pickle

from os.path import isfile, isdir, join, dirname
from tensorflow.keras.activations import softmax, relu
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense
from python_speech_features import *
from scipy.io import wavfile
from ASRModel import ASRModel

MODEL_NAME = 'model.h5'
INFO_NAME = 'info.dictionary'


def gen_sample(data, k):
    for v in data:
        yield v[k]


class ModelPathError(Exception):
    pass


class CNN(ASRModel):

    # constructor
    def __init__(self, model_path: str, wanted_words=None):
        print("CNN constructor")
        self.model_path = model_path if isdir(model_path) else dirname(model_path)
        self.model_id = CNN.get_id_from_path(self.model_path)
        self.test_loss = -1
        self.test_accuracy = -1

        # Load or create model
        if isfile(model_path) or isfile(join(model_path, MODEL_NAME)):                          # load existing model
            self.model = CNN.load_model(model_path)
            cnn_data = CNN.load_data(join(model_path, INFO_NAME))
            self.wanted_words = cnn_data["wanted_words"]
            self.trainset_id = cnn_data["trainset_id"]
            self.testset_id = cnn_data["testset_id"]
            self.info = cnn_data["info"]
            self.model_id = cnn_data["model_id"]
        elif isdir(model_path):                         # create new model
            self.info = {}
            self.trainset_id = ""
            self.testset_id = ""
            self.model = CNN.build_model(len(wanted_words))
            if wanted_words is None:
                self.wanted_words = []
            else:
                self.wanted_words = wanted_words

            # add UNKNOWN label to the dataset if not present
            if dataset_utils.UNKNOWN not in self.wanted_words:
                self.wanted_words.append(dataset_utils.UNKNOWN)

        self.wanted_words.sort()

    def load_dataset(self, trainset, partitions=('train', 'validation', 'test'), t_batch_size=500, v_batch_size=500,
                     test_batch_size=500, val_percentage=10., test_percentage=10., unknown_percentage=10.):

        if isdir(trainset):
            x_train, y_train, x_val, y_val, x_test, y_test, info = \
                dataset_utils.load_dataset(trainset, val_percentage=val_percentage, test_percentage=test_percentage)

            g_train = dataset_utils.dataset_generator(x_train, y_train, self.info, self.wanted_words,
                                                      batch_size=t_batch_size, tot_size=-1,
                                                      unknown_percentage=unknown_percentage, balanced=True)
            g_val = dataset_utils.dataset_generator(x_val, y_val, self.info, self.wanted_words,
                                                    batch_size=v_batch_size, tot_size=-1,
                                                    unknown_percentage=unknown_percentage)
            g_test = dataset_utils.dataset_generator(x_test, y_test, self.info, self.wanted_words,
                                                     batch_size=test_batch_size, tot_size=-1,
                                                     unknown_percentage=unknown_percentage, balanced=True)

            self.info.update(info)
        else:
            import tensorflow_datasets as tfds

            ds_train, info_train = tfds.load('speech_commands', split=tfds.Split.TRAIN, batch_size=t_batch_size,
                                             with_info=True)
            ds_val, info_val = tfds.load('speech_commands', split=tfds.Split.VALIDATION, batch_size=v_batch_size,
                                         with_info=True)
            ds_test, info_test = tfds.load('speech_commands', split=tfds.Split.TEST, batch_size=test_batch_size,
                                          with_info=True)

            self.info.update(info_train)
            self.info.update(info_val)
            self.info.update(info_test)

            g_train = tfds.as_numpy(ds_train)
            g_val = tfds.as_numpy(ds_val)
            g_test = tfds.as_numpy(ds_test)

        # TODO: len(self..wanted_words) is not supported for tfds in batch_preprocessing_gen
        xy_train = self.batch_preprocessing_gen(g_train, ('audio', 'label'), len(self.wanted_words))
        xy_val = self.batch_preprocessing_gen(g_val, ('audio', 'label'), len(self.wanted_words))
        xy_test = self.batch_preprocessing_gen(g_test, ('audio', 'label'), len(self.wanted_words))

        out = []
        if 'train' in partitions:
            out.append(xy_train)
        if 'validation' in partitions:
            out.append(xy_val)
        if 'test' in partitions:
            out.append(xy_test)

        return out

    def preprocess(self, audio):
        """
        from an input path, load the single audio and return preprocessed audio
        which will be the input of the net
        :param audio: input audio path to preprocess
        :return: the preprocessed audio, the model input
        """

        # print("CNN preprocess")
        if isinstance(audio, str) and isfile(audio):
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
                       preemph=0.97, winfunc=lambda x: np.ones((x,))).reshape((99, 13, 1))

    def batch_preprocessing_gen(self, mnist_val, k_list, ww_size):
        for sample in mnist_val:
            batch = sample[k_list[0]]
            labels = sample[k_list[1]]
            preprocessed_batch = np.array([self.preprocess(data) for data in batch])
            preprocessed_label = np.array(
                [np.concatenate((np.zeros(l), np.array([1.0]), np.zeros(ww_size-l-1))) for l in labels])
            # print(preprocessed_label)
            yield preprocessed_batch, preprocessed_label

    @staticmethod
    def build_model(output_shape, input_shape=(99, 13, 1)):
        """
        Create the model structure with the parameters specified
        :return:
        """
        # add layers [! input shape must be (28,28,1) !]
        model = Sequential()
        model.add(Conv2D(64, kernel_size=3, activation=relu, input_shape=input_shape))
        model.add(Conv2D(32, kernel_size=3, activation=relu))
        model.add(Flatten())
        model.add(Dense(output_shape, activation=softmax))  # 11 nodes at output layer (can be changed)
        # self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.SGD(lr=0.01),
                      metrics=['accuracy'])
        # my_optimizers = [keras.optimizers.Adam(), keras.optimizers.SGD(nesterov=True)]
        # model.compile(optimizer=[my_optimizers], loss='categorical_crossentropy', metrics=['accuracy'])

        print("CNN build_model")
        return model

    def train(self, trainset, epochs=10, steps_per_epoch=10, validation_steps=3,
              partitions=('train', 'validation', 'test'), t_batch_size=500, v_batch_size=500, test_batch_size=500,
              val_percentage=10., test_percentage=10., unknown_percentage=10.):
        """
        Train the builded model in the input dataset specified in the
        :return: the id of the builded model, useful to get the .h5 file
        """

        xy_train, xy_val = self.load_dataset(trainset, partitions=('train', 'validation'), t_batch_size=t_batch_size,
                                             v_batch_size=v_batch_size, test_batch_size=test_batch_size,
                                             val_percentage=val_percentage, test_percentage=test_percentage,
                                             unknown_percentage=unknown_percentage)
        self.model.fit(x=xy_train, epochs=epochs, verbose=2, steps_per_epoch=steps_per_epoch,
                       validation_steps=validation_steps,
                       validation_data=xy_val, use_multiprocessing=False)
        print("CNN train")

    @staticmethod
    def get_id_from_path(model_path: str):
        if model_path.endswith(MODEL_NAME):
            model_id = model_path.split(os.sep)[-2]
        elif isdir(model_path):
            model_id = model_path.split(os.sep)[-1]
        else:
            raise ModelPathError("Invalid model path: {}".format(model_path))
        return model_id

    @staticmethod
    def load_model(model_path: str) -> ASRModel:
        """
        load a pretrained model from the specified path
        :param model_path:
        :return:
        """
        # TODO test this method
        if model_path.endswith(MODEL_NAME):
            model = keras.models.load_model(model_path)
        elif isdir(model_path) and isfile(join(model_path, MODEL_NAME)):
            model = keras.models.load_model(join(model_path, MODEL_NAME))
        else:
            raise ModelPathError("Invalid model path: {}".format(model_path))

        print("CNN load_model in {}".format(model_path))
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def save_model(self):
        """
        Save the current model in the specified path
        :param path:
        :return:
        """

        self.model.save(join(self.model_path, MODEL_NAME), overwrite=True)
        self.save_data()
        print("CNN save_model {}".format(self.model_path))

    def save_data(self):
        cnn_data = {"info": self.info,
                    "wanted_words": self.wanted_words,
                    "model_id": self.model_id,
                    "model_path": self.model_path,
                    "trainset_id": self.trainset_id,
                    "testset_id": self.testset_id,
                    "model_type": "CNN",
                    "test_loss": self.test_loss,
                    "test_accuracy": self.test_accuracy
                    }
        with open(join(self.model_path, INFO_NAME), 'wb') as info_file:
            pickle.dump(cnn_data, info_file)

    @staticmethod
    def load_data(path: str):
        with open(path, 'rb') as data_file:
            cnn = pickle.load(data_file)
        return cnn

    def test(self, testset_path: str):
        """
        Test the trained model, with
        :return:
        """

        xy_test = self.load_dataset(testset_path, partitions='test', test_batch_size=500, val_percentage=10.,
                                    test_percentage=10., unknown_percentage=10.)
        metrics = self.model.evaluate(xy_test[0], steps=4, max_queue_size=10, return_dict=True)
        print("CNN test - {}".format(metrics))
        self.test_loss = metrics['loss']
        self.test_accuracy = metrics['accuracy']
        return metrics
