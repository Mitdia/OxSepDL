import torch
from deepxde import config
from deepxde.utils.external import save_loss_history
import os
from utils.visualiser import plot_all
from options import options
from model import model_multiple_solutions
from input_data_reader import get_oxide_params, read_data
from utils.animator import animate_solution_history, animate_trainable_variables_history
from utils.saver import save_options, save_input_data, archive_important_files, setup_experiment
from utils.funcgen import generate_random_solution

config.set_random_seed(15)
config.set_default_float("float64")
torch.set_default_device('cuda')

experiment_name = "BaseModel7OxidesShHaNewNoIntervalRestrictionOnODEFixedTmaxForIntervalRestrictionLoss"
experiment_dir = setup_experiment(experiment_name)
# params = get_oxide_params("ShHa15", ["Ti3O5", "SiO2", "Al2O3"])
# generated_solutions = [generate_random_solution(params, [10, 10, 20]) for _ in range(5)]
# oxide_params = get_oxide_params("ShHa15", ["Ti3O5", "Al2O3"])
oxide_params = get_oxide_params("ShHaNew", ["MnSiO3", "SiO2", "Al2TiO5", "Al2SiO5", "Al2O3", "MgSiO3", "Mg2SiO4"])
# solution_values_array = [(reference, t_grid) for (reference, t_grid, _) in generated_solutions]
# data_path_array = ["5 random synthetic references with four peaks"]
solution_values_array, data_path_array = read_data("Data/ShHaNewAlignedSubTypes/MNRS-236234", ref_num=6)
num_oxides = len(oxide_params)
num_ref = len(solution_values_array)

model, external_trainable_variables, callbacks = model_multiple_solutions(solution_values_array, oxide_params,
                                                                          options, experiment_dir)
loss_weights = ([options["res_loss_weight"]] * num_oxides +
                [options["tbeg_loss_weight"]] * num_oxides +
                [options["ref_loss_weight"] / num_ref, options["mpeak_loss_weight"] / num_ref] * num_ref)
model.compile("adam", lr=options["learning_rate"],
              # loss=["MSE"] * (num_oxides * 2) + ["mse"] * (num_ref * 2),
              loss_weights=loss_weights,
              external_trainable_variables=external_trainable_variables,
              decay=options["decay"])
loss_history, train_state = model.train(iterations=options["iter_num"], callbacks=callbacks)

model.save(os.path.join("Models", experiment_name))

plot_all(model, oxide_params, solution_values_array, loss_history,
         external_trainable_variables, options, experiment_dir)
animate_solution_history(os.path.join(experiment_dir, "movie.gif"), oxide_params, callbacks[-1])
animate_trainable_variables_history(experiment_dir, os.path.join(experiment_dir, "trainable_variables_history.gif"),
                                    oxide_params, options)
save_options(options, os.path.join(experiment_dir, "parameters.txt"))
save_input_data(oxide_params, data_path_array, os.path.join(experiment_dir, "input_data.txt"))
save_loss_history(loss_history, os.path.join(experiment_dir, "LossHistory.txt"))
archive_important_files(experiment_dir)
