import argparse
import os
import json
import random
import string
import shutil
import platform

from typing import Dict
from CNN import CNN
from HMM import HMM
from os.path import join

# filesystem directory, create dir if does not exist
JSON_PATH = "json"

if os.path.isdir("/nfsd"):  # the program is running in the cluster
    TRAIN_PATH = "/nfsd/hda/DATASETS/"
    MACHINE = "blade"
elif os.path.isdir("trainset"):  # the program is not running in the cluster
    TRAIN_PATH = "trainset"
    MACHINE = platform.uname()[1]  # 'jarvis-vivobooks15'
TEST_PATH = "test"
MODEL_PATH = "model"
RES_PATH = "res"

# list
SUPPORTED_ACTION = ["train", "test", "rtasr"]
SUPPORTED_MODEL = ["CNN", "HMM"]


def random_string(string_length=3):
    return ''.join(random.choice(string.ascii_lowercase) for i in range(string_length))


def load_json(json_path):
    with open(json_path) as f:
        data = json.load(f)
    return data


def get_new_model_id(model_base_name: str):
    """
    look in the file system for a new valid model id, attach a random string and create the directory
    for the model
    :param model_base_name: is the name used to identify the trained model
    :return:
    """
    new_model_id = model_base_name + '_' + random_string()
    out_path = join(MODEL_PATH, new_model_id)
    while os.path.exists(out_path):
        new_model_id = model_base_name + '_' + random_string()
        out_path = join(MODEL_PATH, new_model_id)
    os.makedirs(out_path)
    return new_model_id


def check_model_id(model_id: str) -> bool:
    return os.path.exists(join(MODEL_PATH, model_id))


def del_model(model_id):
    """
    delete the model from filesystem if exist
    :param model_id: the id of the model
    :return:
    """
    dir_path = join(MODEL_PATH, model_id)
    if os.path.exists(dir_path):
        print("WARNING: deleting: {}".format(dir_path))
        shutil.rmtree(dir_path)
    else:
        print("WARNING: del_model: The directory does not exist: {}".format(dir_path))


def train(params: Dict):
    """
    build an asrmodel with the parameter in the json file and train it, than free the memory
    :param params: name of the file
    :return:
    """
    assert "model_type" in params, "model_type is not specified"
    assert params["model_type"] in SUPPORTED_MODEL, \
        "model_type not supported: {}, try with {}".format(params["model_type"], str(SUPPORTED_MODEL))
    assert "trainset_id" in params, "trainset_id is not specified"

    trainset_path = join(TRAIN_PATH, params["trainset_id"])

    if "set_model_name" in params:  # specify a string to identify the model
        model_id = get_new_model_id(params["set_model_name"])
    else:
        model_id = get_new_model_id(params["structure_id"])

    if params["model_type"] == "CNN":
        asrmodel = CNN(join(MODEL_PATH, model_id), input_param=params)
    elif params["model_type"] == "HMM":
        asrmodel = HMM(join(MODEL_PATH, model_id))
    else:
        # should never go here
        raise AssertionError("model_type not recognised: {} check {}".format(params["model_type"], SUPPORTED_MODEL))

    asrmodel.train(trainset_path)
    asrmodel.save_model()
    del asrmodel  # free memory
    return model_id


def test(params: Dict):
    """
    load the model from filesystem, test it, than free the memory
    :param params: a distionary with param-value resp as key-value pairs
    :return:
    """
    assert "model_id" in params, "model_id is not specified"
    assert os.path.exists(join(MODEL_PATH, params["model_id"])), \
        "specified model_id does not exist {}".format(params["model_id"])
    assert "testset_id" in params

    model_path = join(MODEL_PATH, params["model_id"])
    if params["model_type"] == "CNN":
        model = CNN(model_path, params)
    elif params["model_type"] == "HMM":
        model = HMM(model_path)
    else:
        # should never go here
        raise AssertionError("model_type not recognised: {} check {}".format(params["model_type"], SUPPORTED_MODEL))

    metrics = model.test(join(TRAIN_PATH, params["testset_id"]))
    model.save_data()
    res_path = save_results(metrics, params["model_id"])
    del model  # free memory


def save_results(metrics, model_id):
    out_path = join(TEST_PATH, model_id + '_' + random_string(3) + '.json')
    while os.path.exists(out_path):
        out_path = join(TEST_PATH, model_id + '_' + random_string(3) + '.json')

    with open(out_path, "w") as test_file:
        json.dump(metrics, test_file, indent=2)

    return out_path


def real_time_asr(params: Dict):
    """
    load the trained model from filesystem, start a session for real time ASR
    :return:
    """
    assert "model_id" in params


def main(action, params, multi_test=None, set_model_name=None):

    params.update({"machine": MACHINE})

    # check extra parameters
    for act in action.split(','):
        assert act in SUPPORTED_ACTION, \
            "specified action is not supported {} try with {}".format(act, str(SUPPORTED_ACTION))

    # add extra parameters
    if set_model_name is not None:
        params["set_model_name"] = set_model_name  # should be use for debugging

    # create dir if does not exist
    if not os.path.exists(JSON_PATH):
        os.makedirs(JSON_PATH)
    if not os.path.exists(TRAIN_PATH):
        os.makedirs(TRAIN_PATH)
    if not os.path.exists(TEST_PATH):
        os.makedirs(TEST_PATH)
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_PATH)
    if not os.path.exists(RES_PATH):
        os.makedirs(RES_PATH)

    if multi_test is not None:
        multi_params = load_json(multi_test)
        for k in multi_params:
            for v in multi_params[k]:
                c_params = params.copy()
                c_params.update({k: v})
                main(action, c_params)
    else:
        # train, test and/or real time ASR ?
        if "train" in action:
            params["model_id"] = train(params)  # set the model_id to eventually test or rtasr the trained model
        if "test" in action:
            test(params)
        if "rtasr" in action:  # real time ASR
            real_time_asr(params)


if __name__ == "__main__":
    # parse input
    parser = argparse.ArgumentParser(description='Process input param')
    parser.add_argument('--action', '-a', type=str, help='Which type of action to perform? (train/test/rtasr')
    parser.add_argument('--model', '-m', type=str, help='Model name to load (not the path)')
    parser.add_argument('--set_model_name', type=str, help='Model name to load (not the path)')
    parser.add_argument('--trainset', '--train', '-tr', type=str, help='train set name to use (not the path)')
    parser.add_argument('--testset', '--test', '-te', type=str, help='test set name to use (not the path)')
    parser.add_argument('--json', type=str, help='test set name to use (not the path)')
    parser.add_argument('--multitest', type=str, help='test set name to use (not the path)')

    args = parser.parse_args()

    print(str(args))

    # check and load input parameters
    assert os.path.exists(args.json), "invalid path for parameters {}".format(args.json)
    assert args.json.endswith(".json"), "--json file format not supported: {}".format(args.json.split(".")[-1])
    params = load_json(args.json)  # a dictionary with all the parameter to train, test or rtasr

    main(args.action, params, multi_test=args.multitest)

    print("Exit correctly")
