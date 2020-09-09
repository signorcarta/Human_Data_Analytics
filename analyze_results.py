import argparse
import json
import os

from os.path import join

from termcolor import colored


def check_acc(min_acc=0.0, max_acc=1.0, n_label=0, structure_id=""):
    model_path = "test"
    print("{:<30}, {:5}, {:<6}, {:<20}, {:<8}, {:9}, {:6}, {:7}".format("test_id", "acc", "m_type", "structure_id", "machine","train(s)", "epochs", "n_label"))
    for res_file in sorted(os.listdir(model_path)):
        res_file_path = join(model_path, res_file)
        if os.path.isfile(res_file_path) and res_file.endswith('.json'):
            # get the results saved in the file
            res_json = {}
            try:
                with open(res_file_path, 'r') as fp:
                    res_json = json.load(fp)  # load the json data of the results
            except json.decoder.JSONDecodeError:
                print(colored("{:<30}, JSONDecodeError".format(res_file[:-5]), "red_1"))
                continue

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

            m_type = param_json["model_type"] if "model_type" in param_json else " "
            m_machine = param_json["machine"][:6] if "machine" in param_json else " "
            m_train_t = param_json["training_time"] if "training_time" in param_json else -1.0
            m_structure_id = param_json["structure_id"] if "structure_id" in param_json else " "
            m_epochs = param_json["epochs"] if "epochs" in param_json else " "
            m_n_labels = len(param_json["wanted_words"]) if "wanted_words" in param_json else " "

            # filter model
            if not (n_label is None or n_label == m_n_labels or n_label <= 0):
                continue
            if not (structure_id == "" or structure_id == m_structure_id):
                continue

            print(colored("{:<30}, {:.3f}, {:<6}, {:<20}, {:<8}, {:8.1f}, {:6}, {:7}"
                          .format(res_file[:-5], acc, m_type, m_structure_id, m_machine, m_train_t, m_epochs,
                                  m_n_labels), color))


if __name__ == "__main__":
    # parse input
    parser = argparse.ArgumentParser(description='Process input param')
    parser.add_argument('--action', '-a', type=str, help='Which type of action to perform? (train/test/rtasr)')
    parser.add_argument('--min_acc', type=float, help='The min acc of the model to show')
    parser.add_argument('--max_acc', type=float, help='The max acc of the model to show')
    parser.add_argument('--n_label', type=int, help='The number of labels of the printed models')
    parser.add_argument('--structure_id', type=str, help='The structure_id of the printed models')

    args = parser.parse_args()

    print(str(args))

    if args.action == "check_acc":
        check_acc(min_acc=args.min_acc, max_acc=args.max_acc, n_label=args.n_label, structure_id=args.structure_id)
