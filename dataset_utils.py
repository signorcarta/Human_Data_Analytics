import hashlib
import os
import random
import re
import numpy as np

from os.path import join, isdir
from scipy.io.wavfile import read, write
from pathlib import Path

MAX_NUM_WAVS_PER_CLASS = 2**27 - 1  # ~134M
UNKNOWN = '_unknown_'


def which_set(filename, validation_percentage, testing_percentage):
    """Determines which data partition the file should belong to.

    We want to keep files in the same training, validation, or testing sets even
    if new ones are added over time. This makes it less likely that testing
    samples will accidentally be reused in training when long runs are restarted
    for example. To keep this stability, a hash of the filename is taken and used
    to determine which set it should belong to. This determination only depends on
    the name and the set proportions, so it won't change as other files are added.

    It's also useful to associate particular files as related (for example words
    spoken by the same person), so anything after '_nohash_' in a filename is
    ignored for set determination. This ensures that 'bobby_nohash_0.wav' and
    'bobby_nohash_1.wav' are always in the same set, for example.

    Args:
    filename: File path of the data sample.
    validation_percentage: How much of the data set to use for validation (from 0 to 100).
    testing_percentage: How much of the data set to use for testing (from 0 to 100).

    Returns:
    String, one of 'training', 'validation', or 'testing'.
    """
    base_name = os.path.basename(filename)
    # We want to ignore anything after '_nohash_' in the file name when
    # deciding which set to put a wav in, so the data set creator has a way of
    # grouping wavs that are close variations of each other.
    hash_name = re.sub(r'_nohash_.*$', '', base_name)
    # This looks a bit magical, but we need to decide whether this file should
    # go into the training, testing, or validation sets, and we want to keep
    # existing files in the same set even if more files are subsequently
    # added.
    # To do that, we need a stable way of deciding based on just the file name
    # itself, so we do a hash of that and then use that to generate a
    # probability value that we use to assign it.
    hash_name_hashed = hashlib.sha1(hash_name.encode('utf-8')).hexdigest()
    percentage_hash = ((int(hash_name_hashed, 16) %
                        (MAX_NUM_WAVS_PER_CLASS + 1)) *
                       (100.0 / MAX_NUM_WAVS_PER_CLASS))
    if percentage_hash < validation_percentage:
        result = 'validation'
    elif percentage_hash < (testing_percentage + validation_percentage):
        result = 'testing'
    else:
        result = 'training'
    return result


def load_dataset(dataset_path, val_percentage=10., test_percentage=10.):
    """
    Return a list of dataset for training, validation and testing
    :param dataset_path: the path of the dir within the directory named as the label of the contained samples
    :param val_percentage: the percentage of validation desired
    :param test_percentage: the percentage of testing desired
    :return:
    """
    x_train, y_train, x_val, y_val, x_test, y_test = [], [], [], [], [], []

    info = {
            "dataset_path": dataset_path,
            "tot_sample": 0,
            "discarded_sample": 0,
            "labels": [],
            "counters": {}
            }

    # load all the labels
    for lab in os.listdir(dataset_path):
        if isdir(join(dataset_path, lab)):
            info["labels"].append(lab)
            info["counters"][lab] = 0

    # load all path, input:
    path_list = []
    for label_dir in os.listdir(dataset_path):
        if isdir(join(dataset_path, label_dir)):
            for file in os.listdir(join(dataset_path, label_dir)):
                # filter all file that are not .wav and with duration different of 1s (32044 bytes)
                if file.endswith(".wav") and Path(join(dataset_path, label_dir, file)).stat().st_size == 32044:
                    path_list.append(join(label_dir, file))
                    info["tot_sample"] += 1
                    info["counters"][label_dir] += 1
                else:
                    info["discarded_sample"] += 1

    # shuffle
    random.shuffle(path_list)

    # split train validation and test
    for sample in path_list:
        data = os.path.basename(sample)
        label = sample.split(os.sep)[-2]
        if which_set(sample, val_percentage, test_percentage) == 'training':
            x_train.append(data)
            y_train.append(label)
        elif which_set(sample, val_percentage, test_percentage) == 'validation':
            x_val.append(data)
            y_val.append(label)
        elif which_set(sample, val_percentage, test_percentage) == 'testing':
            x_test.append(data)
            y_test.append(label)
        else:
            raise Exception("which_set fail! Debug the method.")

    return x_train, y_train, x_val, y_val, x_test, y_test, info


def dataset_generator(x_, y_, info, wanted_words, batch_size=1000, unknown_percentage=10., tot_size=-1, balanced=False):
    """
    This method select the samples for train, validation and test set batches. Moreover it read the audio and the resp
    label. It need the wanted_words list to differentiate sample with label that are not in wanted_words.

    :param x_: the sample of the dataset
    :param y_: the resp labels of the sample
    :param info: contain dataset_path, tot_sample, counters for each label
    :param wanted_words: the list of wanted words of the model
    :param batch_size: the size of each yielded batch
    :param unknown_percentage: the percentage of unknown samples added to each batch
    :param tot_size: used to set a limit to the batch size
    :param balanced: boolean, if True, each yielded batch has a balanced number of samples for each label in wanted
                        words, otherwise each sample of the dataset is added to the batch.
    :return: a generator that yield one batch at a time with two components: 'audio' and 'label'.
    """

    # adjust tot_size (from 1 to |x_|) and batch_size (from 1 to tot_size)
    if tot_size <= 0 or tot_size > len(x_):
        tot_size = len(x_)
    if batch_size <= 0 or batch_size > tot_size:
        batch_size = tot_size

    # check if all label are available in the dataset
    for label in wanted_words:
        if label != UNKNOWN and label not in y_:
            raise Exception("The specified label '{}' is not available in the dataset.".format(label))

    # add UNKNOWN label to the dataset if not present
    if unknown_percentage > 0.0 and UNKNOWN not in wanted_words:
        wanted_words.append(UNKNOWN)

    # alphabetically sort all the label
    wanted_words.sort()

    # calculate the max number of samples for each label
    l_percentage = (100 - unknown_percentage)/(len(wanted_words) - 1)  # -1 is because 'unknown' in ww,
    max_size = {}
    for label in wanted_words:
        if label == UNKNOWN:
            max_size[UNKNOWN] = min(int(unknown_percentage*batch_size/100)+1,
                                    info['tot_sample'] - sum([info['counters'][label_] if UNKNOWN != label_ else 0 for label_ in wanted_words]))
        else:
            max_size[label] = min(int(l_percentage*batch_size/100)+1, info['counters'][label])

    sample_counter = {label: 0 for label in wanted_words}

    # max_iterations = int(tot_size/batch_size)+1
    inner_index = 0
    round_ = 0  # incremented each time inner_index >= len(x_)
    step = 0
    while True:  # the generator can generate batch forever, cycling the whole dataset
        xy_numpy = {'audio': [], 'label': []}
        batch_index = 0
        while batch_index < batch_size:
            if inner_index >= len(x_):  # the entire dataset has been consumed, restart from the beginning
                print("Complete a tour of the whole dataset.")
                round_ += 1
                inner_index = 0  # TODO: is it the best choise? Should the list be shuffled?
                if not balanced:
                    break

            label = y_[inner_index] if y_[inner_index] in wanted_words else UNKNOWN  # the label of the current sample

            # add the sample to the yield batch if needed
            if balanced:
                # evaluate if this label has too much samples in the current batch
                if sample_counter[label] < max_size[label]:
                    sample_counter[label] += 1
                    fs, data = read(join(info["dataset_path"], y_[inner_index], x_[inner_index]))
                    xy_numpy['audio'].append(data)
                    label_index = wanted_words.index(label)
                    xy_numpy['label'].append(label_index)
                    batch_index += 1
            else:
                sample_counter[label] += 1
                fs, data = read(join(info["dataset_path"], y_[inner_index], x_[inner_index]))
                xy_numpy['audio'].append(data)
                label_index = wanted_words.index(label)
                xy_numpy['label'].append(label_index)
                batch_index += 1

            inner_index += 1
        if len(xy_numpy['label']) == 0:  # happen when complete a tour of the whole dataset and at the same time,
            continue                     # complete the batch. It will restart the dataset without yielding a void batch

        step += 1
        print("round {:3.0f}, step {} , examined {}/{} , batch_size {} ".format(round_, step, inner_index, len(x_),
                                                                                len(xy_numpy['label'])))
        yield xy_numpy
        sample_counter = {label: 0 for label in wanted_words}  # clean sample counter
    print("dataset_generator end!")  # it should never end


if __name__ == "__main__":
    file_path = "trainset/speech_commands_v0.02/bird/9ff2d2f4_nohash_0.wav"
    print(which_set(file_path, 20., 30.))

    dataset_path = "trainset/speech_commands_v0.02"
    x_train, y_train, x_val, y_val, x_test, y_test, info = load_dataset(dataset_path, val_percentage=10.,
                                                                        test_percentage=10.)
    for x_ in x_train[:10]:
        print(x_)
