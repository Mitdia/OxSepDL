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

config.set_random_seed(57)
config.set_default_float("float64")
torch.set_default_device('cuda')


def parse_record(rec):
    rec = rec.strip("'")
    rec = rec.replace(",", "")
    rec = rec.replace("[", "")
    rec = rec.replace("]", "")
    return [float(value) for value in rec.split()]


with open("Experiments/BaseModel8OxidesCorrectTMelting-1/variables.dat") as variables_history_file:
    latest_vars = [parse_record(record)[1:] for record in variables_history_file.read().split("\n")][-2]


experiment_dir = setup_experiment("fineTune8Oxides")
oxide_params = get_oxide_params("ShHa15", ["MnO", "MnSiO3", "SiO2", "Al2TiO5", "Al2O3", "MgAl2O4", "CaAl4O7", "Mg2SiO4"])
_, _ = read_data(ref_num=1)
solution_values_array, data_path_array = read_data("Data/ShHa2Unaligned", ref_num=1)

num_oxides = len(oxide_params)
num_ref = len(solution_values_array)

model, external_trainable_variables, callbacks = model_multiple_solutions(solution_values_array, oxide_params,
                                                                          options, experiment_dir,
                                                                          trainable_vars_init=latest_vars)
loss_weights = ([options["res_loss_weight"]] * num_oxides +
                [options["tbeg_loss_weight"]] * num_oxides +
                [options["ref_loss_weight"] * 1e+1 / num_ref, options["mpeak_loss_weight"] * 1e+1 / num_ref] * num_ref)

# loss_weights = [1e+4] * num_oxides + [1e+2] * num_oxides + [1e+ / num_ref, 0 / num_ref] * num_ref
model.compile("adam", lr=5e-6,
              loss_weights=loss_weights,
              external_trainable_variables=external_trainable_variables,
              decay=options["decay"])


# model.restore("Models/BaseModelStrictESoftTmax-200000.pt")
checkpoint = torch.load("Models/BaseModel8OxidesCorrectTMelting-200000.pt")
model.net.load_state_dict(checkpoint["model_state_dict"])
loss_history, train_state = model.train(iterations=10000, callbacks=callbacks, display_every=100)

plot_all(model, oxide_params, solution_values_array, loss_history,
         external_trainable_variables, options, experiment_dir)
animate_solution_history(os.path.join(experiment_dir, "movie.gif"), oxide_params, callbacks[-1])
animate_trainable_variables_history(experiment_dir, os.path.join(experiment_dir, "trainable_variables_history.gif"),
                                    oxide_params, options)
save_options(options, os.path.join(experiment_dir, "parameters.txt"))
save_input_data(oxide_params, data_path_array, os.path.join(experiment_dir, "input_data.txt"))
save_loss_history(loss_history, os.path.join(experiment_dir, "LossHistory.txt"))
archive_important_files(experiment_dir)
