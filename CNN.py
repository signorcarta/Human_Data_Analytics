from ASRModel import ASRModel
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras.activations import relu, softmax

import os


class CNN(ASRModel):

    # constructor
    def __init__(self, model_id: str):
        print("CNN constructor")
        self.model_id = model_id

        # Create model
        self.model = Sequential()

    def preprocess(self, audio_path: str):
        """
        from an input path, load the single audio and return preprocessed audio
        which will be the input of the net
        :param audio_path: input audio path to preprocess
        :return: the preprocessed audio, the model input
        """
        print("CNN preprocess")

    def build_model(self):
        """
        Create the model structure with the parameters specified in the constructor
        :return:
        """
        # add layers [! input shape must be (28,28,1) !]
        self.model.add(Conv2D(64, kernel_size=3, activation=relu, input_shape=(28, 28, 1)))
        self.model.add(Conv2D(32, kernel_size=3, activation=relu))
        self.model.add(Flatten())
        self.model.add(Dense(10, activation=softmax))  # 10 nodes at output layer (can be changed)
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        print("CNN build_model")

    def train(self, trainset_path: str):
        """
        Train the builded model in the input dataset specified in the
        :return: the id of the builded model, useful to get the .h5 file
        """
        self.model.fit(trainset_path, trainset_path, epochs=3) #validation_data=(X_test, y_test), epochs=3)

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

        self.model.save(path, overwrite=True, include_optimizer=True, save_format=None, signatures=None, options=None)
        print("CNN save_model")

    def test(self, testset_path: str):
        """
        Test the trained model, with
        :return:
        """

        self.model.predict(testset_path)
        print("CNN test")
