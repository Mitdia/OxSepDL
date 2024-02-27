import torch
from deepxde import config
import os
from utils.visualiser import plot_all
from options import options
from model import model_multiple_solutions
from imput_data_reader import get_oxide_params, read_data
from utils.animator import animate_solution_history, animate_trainable_variables_history
from utils.saver import save_options, save_input_data, archive_important_files, setup_experiment

config.set_random_seed(42)
config.set_default_float("float64")
torch.set_default_device('cuda')

experiment_dir = setup_experiment()
solution_values_array, data_path_array = read_data()
oxide_params = get_oxide_params("ShHa15", ["SiO2", "Al2O3"])
num_oxides = len(oxide_params)
num_ref = len(solution_values_array)

model, external_trainable_variables, callbacks = model_multiple_solutions(solution_values_array, oxide_params,
                                                                          options, experiment_dir)
loss_weights = ([options["res_loss_weight"]] * num_oxides +
                [options["tbeg_loss_weight"]] * num_oxides +
                [options["ref_loss_weight"] / num_ref, options["mpeak_loss_weight"] / num_ref] * num_ref)
model.compile("adam", lr=options["learning_rate"], loss_weights=loss_weights,
              external_trainable_variables=external_trainable_variables,
              decay=options["decay"])
loss_history, train_state = model.train(iterations=options["iter_num"], callbacks=callbacks)


plot_all(model, oxide_params, solution_values_array, loss_history,
         external_trainable_variables, options, experiment_dir)
animate_solution_history(os.path.join(experiment_dir, "movie.gif"), oxide_params, callbacks[-1])
animate_trainable_variables_history(experiment_dir, os.path.join(experiment_dir, "trainable_variables_history.gif"),
                                    oxide_params, options)
save_options(options, os.path.join(experiment_dir, "parameters.txt"))
save_input_data(oxide_params, data_path_array, os.path.join(experiment_dir, "input_data.txt"))
archive_important_files(experiment_dir)
