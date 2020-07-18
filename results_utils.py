import numpy as np
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import main
import os
import json
import sys

from os.path import isdir, join, isfile


def cm_plot(cm, wanted_words, model_id, save=False):
    df_cm = pd.DataFrame(cm, index=wanted_words, columns=wanted_words)
    plt.figure(figsize=(11.4, 9))
    sn.heatmap(df_cm, annot=True)
    plt.title(model_id)
    if save:
        plt.savefig(join("res", model_id + "_cm"))
    plt.show()


def load_model_res_and_data(model_id):
    print("{} results and parameters".format(model_id))
    res = []
    for res_id in os.listdir(main.TEST_PATH):
        if model_id in res_id:
            with open(join(main.TEST_PATH, res_id), "r") as f:
                res.append(json.load(f))
            res[-1].update({"test_id": res_id.split(".")[0]})

    model_dir = join(main.MODEL_PATH, model_id)
    param = {}
    if isfile(join(model_dir, "param.json")):
        with open(join(model_dir, "param.json")) as f:
            param.update(json.load(f))
    else:
        print("{} : no parameter saved".format(model_id))

    return param, res


def all_models_results(save=False):
    for model_id in os.listdir(main.MODEL_PATH):
        single_model_results(model_id, save=save)


def single_model_results(model_id, save=False):
    param, res = load_model_res_and_data(model_id)
    if res != [] and param != {}:
        for r in res:
            if "confusion_matrix" in r and "wanted_words" in param and "test_id" in r:
                cm_plot(r["confusion_matrix"], param["wanted_words"], r["test_id"], save=save)
            else:
                print("model {} has no confusion matrix or wanted world.".format(model_id))


if __name__ == "__main__":
    if sys.argv[1] in ("True", "False"):
        all_models_results(save=sys.argv[1])
    else:
        for test in os.listdir(main.TEST_PATH):
            if sys.argv[1] in test:
                single_model_results(sys.argv[1])
        print("")

