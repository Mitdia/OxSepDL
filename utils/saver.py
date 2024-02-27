import json
import copy
import zipfile
import os
from datetime import datetime


def setup_experiment(experiment_name=None):
    if experiment_name is None:
        current_date = datetime.now()
        formatted_date = current_date.strftime("%B-%d")
        experiment_name = formatted_date
    directory_name = os.path.join("Experiments", experiment_name)
    i = 0
    new_directory_name = directory_name
    while os.path.exists(new_directory_name):
        i += 1
        new_directory_name = f"{directory_name}-{i}"
    os.makedirs(new_directory_name)
    return new_directory_name


def save_options(options, filename):
    options_to_save = options.copy()
    for key, value in options_to_save.items():
        if callable(value):
            options_to_save[key] = value.__doc__.split("\n")[-2]

    with open(filename, 'w') as params_file:
        json.dump(options_to_save, params_file, indent=0)


def save_input_data(oxide_params, data_path_array, filename):
    input_data = copy.deepcopy(oxide_params)
    input_data["references"] = list(data_path_array)
    with open(filename, 'w') as params_file:
        json.dump(input_data, params_file, indent=0)


def archive_important_files(experiment_path):
    to_save = [
        "ODELoss.png",
        "IdealFunctionsForPredictedVariables.png",
        "Result.png",
        "TbegLoss.png",
        "ReferenceLoss.png",
        "MaxPeakLoss.png",
        "movie.gif",
        "trainable_variables_history.gif",
        "test.dat",
        "variables.dat",
        "train.dat",
        "parameters.txt",
        "LossHistory.png",
        "input_data.txt",
        "SolutionHistory_x.txt",
        "SolutionHistory_y.txt",
    ]

    with zipfile.ZipFile(os.path.join(experiment_path, "ImportantInfo.zip"), 'w') as zipf:
        for file_name in to_save:
            path = os.path.join(str(experiment_path), file_name)
            if os.path.exists(path):
                zipf.write(path)
