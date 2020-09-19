# HumanDataAnalitycs
This is the repo to manage the HDA course project. The project will be about Automatic Speech Recognition

## Get started
Using conda it is possible to create an environment which can run main.py
```bash
conda env create -f environment.yml
```
main.py program can compute three different so called actions: `train` a new CNN, `test` an existing one or run a real time prediction (`rtasr`) using the microphone if avaiable. Each of them require a configuration file which must be specified with `--json` parameters. The repository is provided of all the json template needed to run each actions.
Here an example of command to execute a train:
```bash
python main.py --action train --json json/train_param_template.json
```
a test:
```bash
python main.py --action test --json json/test_param_template.json
```
or a real time prediction:
```bash
python main.py --action rtasr --model_id <model_id>
```
