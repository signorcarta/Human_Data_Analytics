import argparse
import json
import os
import time

from os.path import join

from termcolor import colored

MODEL_PATH = "test"
DEFAULT_PARAM_LIST = (  # the list of parameter to print
    "test_id", "m_type", "structure_id", "machine",
    "preprocess_type", #"winlen", "winstep", "numcep", "nfilt",
    "n_label", "epochs", "tot_sample",
    #"train(s)", "prep(s)", "load(s)",
    "opt", "loss", "acc", "date"
)
STR_TITLE_FORMAT = {  # the format for each parameter in the title line
    "test_id": "{:<30}",
    "preprocess_type": "{:<15}",
    "winlen": "{:<7}",
    "winstep": "{:<7}",
    "numcep": "{:<7}",
    "nfilt": "{:<7}",
    "acc": "{:5}",
    "m_type": "{:<6}",
    "structure_id": "{:<20}",
    "machine": "{:<8}",
    "train(s)": "{:8}",
    "prep(s)": "{:7}",
    "load(s)": "{:7}",
    "epochs": "{:6}",
    "n_label": "{:7}",
    "opt": "{:<5}",
    "loss": "{:24}",
    "tot_sample": "{:9}",
    "date": "{:<15}",
}
STR_DATA_FORMAT = {  # the format for each parameter in the data line
    "test_id": "{:<30}",
    "preprocess_type": "{:<15}",
    "winlen": "{:7.1f}",
    "winstep": "{:7.1f}",
    "numcep": "{:7}",
    "nfilt": "{:7}",
    "acc": "{:.3f}",
    "m_type": "{:<6}",
    "structure_id": "{:<20}",
    "machine": "{:<8}",
    "train(s)": "{:8.1f}",
    "prep(s)": "{:7.1f}",
    "load(s)": "{:7.1f}",
    "epochs": "{:6}",
    "n_label": "{:7}",
    "opt": "{:<5}",
    "loss": "{:24}",
    "tot_sample": "{:9}",
    "date": "{:<15}",
}
# latex format = ["\\item ", " & ", "\\\\"]
# terminal format = ["", ", ", ""]
SEP = [["", ", ", ""], ["", " & ", " \\\\"]][0]  # begin with SEP[0], divide with SEP[1] and end the line with SEP[2]


def check_acc(param_list=DEFAULT_PARAM_LIST, min_acc=0.0, max_acc=1.0, n_label=0, structure_id="", optimizer="",
              epochs=-1, tot_sample=-1, date=""):
    param_value = {}

    # create the title line
    title_formatted_output = SEP[0] + SEP[1].join([STR_TITLE_FORMAT[p] for p in param_list]) + SEP[2]
    title_line = title_formatted_output.format(*param_list)
    print(title_line)

    #
    for res_file in sorted(os.listdir(MODEL_PATH)):
        res_file_path = join(MODEL_PATH, res_file)
        if os.path.isfile(res_file_path) and res_file.endswith('.json'):
            # get the results saved in the file
            res_json = {}
            try:
                with open(res_file_path, 'r') as fp:
                    res_json = json.load(fp)  # load the json data of the results
            except json.decoder.JSONDecodeError:
                print(colored("{:<30}, JSONDecodeError".format(res_file[:-5]), "red_1"))
                continue
            param_value["date"] = time.strftime("%Y/%m/%w %H:%M", time.gmtime(os.path.getmtime(res_file_path)))

            # get the accuracy
            if "test_accuracy" in res_json:
                acc = res_json["test_accuracy"]
            elif "accuracy" in res_json:
                acc = res_json["accuracy"]
            else:
                print(colored("{:<30}, NO accuracy found".format(res_file[:-5]), "red_1"))
                continue


            # filter results
            if not (min_acc <= acc <= max_acc):
                continue

            # set the color of the printed results
            if acc > .8:
                color = "grey"
            elif acc > .6:
                color = "yellow"
            elif acc > .4:
                color = "cyan"
            elif acc > .2:
                color = "magenta"
            else:
                color = "white"

            # get the model name
            model_name = res_file.split('.')[0][:-4]  # <string>_<model_hash:3>_<test_hash:3>

            # print output message
            param_json = {}
            param_file = join("model", model_name, "param.json")
            if os.path.isfile(param_file):  # the model params are saved
                try:
                    with open(param_file, 'r') as param_fp:
                        param_json = json.load(param_fp)  # load the json data of the results
                except json.decoder.JSONDecodeError:
                    print(colored("{:<30}, {:.3f}, JSONDecodeError".format(param_file[:-5], acc), "red_1"))
                    continue
            else:
                print(colored("{:<30}, {:.3f}, NO param.json".format(res_file[:-5], acc), color))
                continue

            if "epochs" in param_json and "steps_per_epoch" in param_json and "t_batch_size" in param_json:
                param_value["tot_sample"] = param_json["epochs"] * param_json["steps_per_epoch"] * param_json["t_batch_size"]

            param_value["test_id"] = res_file[:-5]
            param_value["acc"] = acc
            param_value["m_type"] = param_json["model_type"] if "model_type" in param_json else " "
            param_value["preprocess_type"] = param_json["preprocess_type"] if "preprocess_type" in param_json else " "
            param_value["machine"] = param_json["machine"][:6] if "machine" in param_json else " "
            param_value["train(s)"] = param_json["training_time"] if "training_time" in param_json else -1.0
            param_value["prep(s)"] = param_json["preproces_tot_time"] if "preproces_tot_time" in param_json else -1.0
            param_value["load(s)"] = param_json["load_dataset_time"] if "load_dataset_time" in param_json else -1.0
            param_value["structure_id"] = param_json["structure_id"] if "structure_id" in param_json else " "
            param_value["epochs"] = param_json["epochs"] if "epochs" in param_json else " "
            param_value["n_label"] = len(param_json["wanted_words"]) if "wanted_words" in param_json else " "
            param_value["opt"] = param_json["optimizer"] if "optimizer" in param_json else " "
            param_value["loss"] = param_json["loss"] if "loss" in param_json else " "
            param_value["winlen"] = param_json["winlen"] if "winlen" in param_json else " "
            param_value["winstep"] = param_json["winstep"] if "winstep" in param_json else " "
            param_value["numcep"] = param_json["numcep"] if "numcep" in param_json else " "
            param_value["nfilt"] = param_json["nfilt"] if "nfilt" in param_json else " "

            # filter model
            if not (n_label is None or n_label == param_value["n_label"] or n_label <= 0):
                continue
            if not (structure_id == "" or structure_id == param_value["structure_id"]):
                continue
            if not (optimizer == "" or optimizer == param_value["opt"]):
                continue
            if not (epochs <= 0 or epochs == param_value["epochs"]):
                continue
            if not (tot_sample <= 0 or tot_sample == param_value["tot_sample"]):
                continue
            if not (date == "" or date <= param_value["date"]):
                continue

            # composition of the line within the values of the selected parameter
            data_formatted_output = SEP[0] + SEP[1].join([STR_DATA_FORMAT[p] for p in param_list]) + SEP[2]
            data_line = data_formatted_output.format(*[param_value[p] for p in param_list])
            print(colored(data_line, color))


if __name__ == "__main__":
    # parse input
    parser = argparse.ArgumentParser(description='Process input param')
    parser.add_argument('--action', '-a', type=str, help='Which type of action to perform? (train/test/rtasr)')
    parser.add_argument('--min_acc', type=float, help='The min acc of the model to show')
    parser.add_argument('--max_acc', type=float, help='The max acc of the model to show')
    parser.add_argument('--n_label', type=int, help='The number of labels of the printed models')
    parser.add_argument('--epochs', type=int, help='The number of epochs of the printed models')
    parser.add_argument('--tot_sample', type=int, help='The number of sample used to train the models')
    parser.add_argument('--structure_id', type=str, help='The structure_id of the printed models')
    parser.add_argument('--optimizer', type=str, help='The optimizer used for the train')
    parser.add_argument('--param_list', type=str, help='The list of parameter to print')
    parser.add_argument('--date', type=str, help='The starting date of the model creation')

    args = parser.parse_args()

    print(str(args))

    if args.action == "check_acc":
        param_list = args.param_list.split(",") if args.param_list is not None else DEFAULT_PARAM_LIST
        check_acc(param_list=param_list, min_acc=args.min_acc, max_acc=args.max_acc, n_label=args.n_label,
                  structure_id=args.structure_id, optimizer=args.optimizer, epochs=args.epochs,
                  tot_sample=args.tot_sample, date=args.date)
