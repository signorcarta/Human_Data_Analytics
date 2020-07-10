import numpy as np
import os
import tensorflow.keras as keras
import dataset_utils
import pickle
import json

from os.path import isfile, isdir, join, dirname
from tensorflow.keras.activations import softmax, relu
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense
from python_speech_features import *
from scipy.io import wavfile
from ASRModel import ASRModel

MODEL_NAME = 'model.h5'
INFO_NAME = 'param.json'


# TODO: confusion matrix
# run on blade
# analyze result
# train parameter, test parameter from json

def gen_sample(data, k):
    for v in data:
        yield v[k]


class ModelPathError(Exception):
    pass


class CNN(ASRModel):

    # constructor
    def __init__(self, model_path: str, input_param=None):
        print("CNN constructor")
        # param to identify a cnn model
        self.model_path = model_path if isdir(model_path) else dirname(model_path)
        self.model_id = CNN.get_id_from_path(self.model_path)
        self.machine = input_param['machine']

        # output initialization
        self.out_param = {}

        # Load or create model, wanted_words and info
        if isfile(model_path) or isfile(join(model_path, MODEL_NAME)):      # load existing model
            input_param.update(CNN.load_data(join(model_path, INFO_NAME)))

            # compile model
            self.optimizer = input_param["optimizer"]  # 'adam'
            self.loss = input_param["loss"]  # 'categorical_crossentropy'
            self.metrics = input_param["metrics"]  # ('accuracy')

            self.model = self.load_model(model_path)

            self.wanted_words = input_param["wanted_words"]
            self.numcep = input_param["numcep"]
            self.info = input_param["info"]
        elif isdir(model_path):                                             # create new model
            self.info = {}
            self.wanted_words = input_param["wanted_words"]
            self.numcep = input_param["numcep"]

            self.model = CNN.build_model(len(input_param["wanted_words"]), input_shape=(99, self.numcep, 1))

            # compile model
            self.optimizer = input_param["optimizer"]  # 'adam'
            self.loss = input_param["loss"]  # 'categorical_crossentropy'
            self.metrics = input_param["metrics"]  # ('accuracy')

            # add UNKNOWN label to the dataset if not present
            # TODO: talk about unknown in the report
            if dataset_utils.UNKNOWN not in self.wanted_words:
                self.wanted_words.append(dataset_utils.UNKNOWN)
        self.wanted_words.sort()

        # preprocess param
        self.winlen = input_param["winlen"]  #: 0.025,
        self.winstep = input_param["winstep"]  #: 0.01,
        self.nfilt = input_param["nfilt"]  #: 26,
        self.nfft = input_param["nfft"]  #: 512,
        self.preemph = input_param["preemph"]  #: 0.97,

        # param to build model
        self.structure_id = input_param["structure_id"]  #: "light_cnn",
        self.filters = input_param["filters"]  #: [64, 32],
        self.kernel_size = input_param["kernel_size"]  #: [3, 3],

        # input dataset of the model
        self.trainset_id = input_param["trainset_id"]
        self.testset_id = input_param["testset_id"]

        # train
        self.epochs = input_param["epochs"]  # 10
        self.steps_per_epoch = input_param["steps_per_epoch"]  # 10
        self.validation_steps = input_param["validation_steps"]  # 3

        self.t_batch_size = input_param["t_batch_size"]  # 500
        self.v_batch_size = input_param["v_batch_size"]  # 500
        self.val_percentage = input_param["val_percentage"]  # 10.

        self.test_batch_size = input_param["test_batch_size"]  # 500
        self.test_steps = input_param["test_steps"]  # 4
        self.test_percentage = input_param["test_percentage"]  # 10.

        self.unknown_percentage = input_param["unknown_percentage"]  # 10.

        # test

    def load_dataset(self, trainset, partitions=('train', 'validation')):

        if isdir(trainset):
            x_train, y_train, x_val, y_val, x_test, y_test, info = \
                dataset_utils.load_dataset(trainset, val_percentage=self.val_percentage, test_percentage=self.test_percentage)

            g_train = dataset_utils.dataset_generator(x_train, y_train, self.info, self.wanted_words,
                                                      batch_size=self.t_batch_size, tot_size=-1,
                                                      unknown_percentage=self.unknown_percentage, balanced=True)
            g_val = dataset_utils.dataset_generator(x_val, y_val, self.info, self.wanted_words,
                                                    batch_size=self.v_batch_size, tot_size=-1,
                                                    unknown_percentage=self.unknown_percentage)
            g_test = dataset_utils.dataset_generator(x_test, y_test, self.info, self.wanted_words,
                                                     batch_size=self.test_batch_size, tot_size=-1,
                                                     unknown_percentage=self.unknown_percentage, balanced=True)

            self.info.update(info)
        else:
            import tensorflow_datasets as tfds

            ds_train, info_train = tfds.load('speech_commands', split=tfds.Split.TRAIN, batch_size=self.t_batch_size,
                                             with_info=True)
            ds_val, info_val = tfds.load('speech_commands', split=tfds.Split.VALIDATION, batch_size=self.v_batch_size,
                                         with_info=True)
            ds_test, info_test = tfds.load('speech_commands', split=tfds.Split.TEST, batch_size=self.test_batch_size,
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
        print("CNN build_model")
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

        return model

    def train(self, trainset):
        """
        Train the builded model in the input dataset specified in the
        :return: the id of the builded model, useful to get the .h5 file
        """

        print("CNN train")
        xy_train, xy_val = self.load_dataset(trainset, partitions=('train', 'validation'))
        self.model.fit(x=xy_train, epochs=self.epochs, verbose=2, steps_per_epoch=self.steps_per_epoch,
                       validation_steps=self.validation_steps,
                       validation_data=xy_val, use_multiprocessing=False)

    @staticmethod
    def get_id_from_path(model_path: str):
        if model_path.endswith(MODEL_NAME):
            model_id = model_path.split(os.sep)[-2]
        elif isdir(model_path):
            model_id = model_path.split(os.sep)[-1]
        else:
            raise ModelPathError("Invalid model path: {}".format(model_path))
        return model_id

    def load_model(self, model_path: str) -> ASRModel:
        """
        load a pretrained model from the specified path
        :param model_path:
        :return:
        """
        print("CNN load_model in {}".format(model_path))
        if model_path.endswith(MODEL_NAME):
            model = keras.models.load_model(model_path)
        elif isdir(model_path) and isfile(join(model_path, MODEL_NAME)):
            model = keras.models.load_model(join(model_path, MODEL_NAME))
        else:
            raise ModelPathError("Invalid model path: {}".format(model_path))

        model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)
        return model

    def save_model(self):
        """
        Save the current model in the specified path
        :param path:
        :return:
        """

        print("CNN save_model {}".format(self.model_path))
        self.model.save(join(self.model_path, MODEL_NAME), overwrite=True)
        self.save_data()

    def save_data(self):
        cnn_data = {
                    "machine": self.machine,
                    "info": self.info,
                    "wanted_words": self.wanted_words,
                    "model_id": self.model_id,
                    "model_path": self.model_path,

                    "trainset_id": self.trainset_id,
                    "testset_id": self.testset_id,
                    "model_type": "CNN",
                    "winlen": self.winlen,  #: 0.025,
                    "winstep": self.winstep,  #: 0.01,
                    "numcep": self.numcep,  #: 13,
                    "nfilt": self.nfilt,  #: 26,
                    "nfft": self.nfft,  #: 512,
                    "preemph": self.preemph,  #: 0.97,

                    "structure_id": self.structure_id,  #: "light_cnn",
                    "filters": self.filters,  #: [64, 32],
                    "kernel_size": self.kernel_size,  #: [3, 3],

                    "epochs": self.epochs,
                    "t_batch_size": self.t_batch_size,
                    "steps_per_epoch": self.steps_per_epoch,

                    "v_batch_size": self.v_batch_size,
                    "val_percentage": self.val_percentage,
                    "validation_steps": self.validation_steps,

                    "test_batch_size": self.test_batch_size,
                    "test_steps": self.test_steps,
                    "test_percentage": self.test_percentage,
                    "unknown_percentage": self.unknown_percentage

                    }
        with open(join(self.model_path, INFO_NAME), 'w') as info_file:
            json.dump(cnn_data, info_file, indent=2, sort_keys=True)

    @staticmethod
    def load_data(path: str):
        with open(path, 'r') as data_file:
            cnn = json.load(data_file)
        return cnn

    def test(self, testset_path: str):
        """
        Test the trained model, with
        :return:
        """
        xy_test = self.load_dataset(testset_path, partitions='test')
        if self.machine == 'blade':
            metrics = self.model.evaluate(xy_test[0], steps=self.test_steps, max_queue_size=10, verbose=0)
            metrics = {out: float(metrics[i]) for i, out in enumerate(self.model.metrics_names)}
            print("output metrics: {} ".format(str(metrics)))
        else:
            metrics = self.model.evaluate(xy_test[0], steps=self.test_steps, max_queue_size=10, return_dict=True, verbose=0)
        print("CNN test - {}".format(metrics))
        return metrics
