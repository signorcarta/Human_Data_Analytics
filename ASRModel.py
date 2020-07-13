import abc


class ASRModel:

    @staticmethod
    @abc.abstractmethod
    def preprocess(audio_path):
        """
        from an input path, load the single audio and return preprocessed audio
        which will be the input of the net
        :param audio_path: input audio path to pre process
        :return: the preprocessed audio, the model input
        """
        pass

    @staticmethod
    @abc.abstractmethod
    def build_model(output_shape, input_shape=(99, 13, 1)):
        """
        Create the model structure with the parameters specified
        :return:
        """
        pass

    @abc.abstractmethod
    def train(self, trainset_path: str):
        """
        Train the builded model in the input dataset specified in the
        :return: the id of the builded model, usefull to get the .h5 file
        """
        pass

    @staticmethod
    @abc.abstractmethod
    def load_model(model_path: str):
        """
        load a pretrained model from the specified path
        :param model_path:
        :return:
        """
        pass

    @abc.abstractmethod
    def save_model(self):
        """
        Save the current model in the specified path
        :param path:
        :return:
        """
        pass

    @abc.abstractmethod
    def test(self, testset_path: str):
        """
        Test the trained model, with
        :return:
        """
        pass
