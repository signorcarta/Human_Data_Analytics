from ASRModel import ASRModel

import os


class CNN(ASRModel):

    # constructor
    def __init__(self, model_id: str):
        print("CNN constructor")
        self.model_id = model_id

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
        print("CNN build_model")

    def train(self, trainset_path: str):
        """
        Train the builded model in the input dataset specified in the
        :return: the id of the builded model, usefull to get the .h5 file
        """
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
        print("CNN save_model")

    def test(self, testset_path: str):
        """
        Test the trained model, with
        :return:
        """
        print("CNN test")
