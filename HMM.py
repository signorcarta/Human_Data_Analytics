from ASRModel import ASRModel

import os


class HMM(ASRModel):

    # constructor
    def __init__(self, model_id: str):
        self.model_id = model_id

    def preprocess(self, audio_path: str):
        """
        from an input path, load the single audio and return preprocessed audio
        which will be the input of the net
        :param audio_path: input audio path to preprocess
        :return: the preprocessed audio, the model input
        """
        print("HMM preprocess")

    def build_model(self):
        """
        Create the model structure with the parameters specified in the constructor
        :return:
        """
        # self.model = Sequential()
        print("HMM build_model")

    def train(self, trainset_path: str):
        """
        Train the builded model in the input dataset specified in the
        :return: the id of the builded model, usefull to get the .h5 file
        """
        print("HMM train")

    @staticmethod
    def load_model(model_path: str) -> ASRModel:
        """
        load a pre trained model from the specified path. Must have the same effect as build_model,
        but load saved model from filesystem
        :param model_path: the path of the directory named as model_id
        :return:
        """
        if model_path.endswith(os.sep):  # support path ending with '/' or '\'
            model_path = model_path[:-1]
        assert os.path.isdir(model_path), "model_path is not a dir: {}".format(model_path)
        model_id = os.path.basename(model_path)
        print("HMM load_model {}".format(model_path))
        hmm = HMM(model_id)
        # cnn.graph = load_graph()
        return hmm

    def save_model(self, path: str):
        """
        Save the current model in the specified path
        :param path:
        :return:
        """
        print("HMM save_model")

    def test(self, testset_path: str):
        """
        Test the trained model, with
        :return:
        """
        print("HMM test")