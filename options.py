import torch
from deepxde import config
from functools import partial


def get_k_indirect(e_var, t_max, _, k_init):
    """
    Calculate K from the trainable variables
    __________
    Parameters
    __________
    e_var : double
        trainable variable corresponding to E
    t_max : double
        trainable variable corresponding to temperature of maximum extortion
    _______
    Returns
    _______
    K : double
        K = E / T_max + log(E / T_max ** 2)
    """
    return e_var / t_max + torch.log(e_var / t_max ** 2)


def get_k_direct(_, __, k_var, k_scale, k_init):
    """
    Calculate K from the trainable variables
    __________
    Parameters
    __________
    k_var : double
        trainable variable corresponding to K
    k_scale : double
        hyperparameter corresponding to K
    _______
    Returns
    _______
    K : double
        K = k_init * (10 ** torch.tanh(k_var))
    """
    return k_init * (10 ** torch.tanh(k_var))


def get_e(e_var, e_scale, e_init):
    """
    Calculate E from the trainable variable e_var
    __________
    Parameters
    __________
    e_var : double
        trainable variable corresponding to E
    e_scale : double
        The hyperparameter corresponding to E
    _______
    Returns
    _______
    E : double
        E = 10 ^ (5 + torch.tanh(e_var))
    """
    return e_init * (10 ** torch.tanh(e_var))


def get_t_max(t_max, tmax_scale, t_max_init=None):
    """
    Calculate temperature of maximum extortion from the trainable variable t_max
    __________
    Parameters
    __________
    t_max : double
        trainable variable corresponding to temperature of maximum extortion
    tmax_scale : double
        The hyperparameter corresponding to temperature of maximum extortion;
        It defines how far the maximum temperature can diverge from the initial value
    t_max_init : double
        The initial value of the temperature of maximum extortion
    _______
    Returns
    _______
    T_max : double
        T_max = t_max_init + (torch.tanh(t_max) * tmax_scale)
    """
    return t_max_init + (torch.tanh(t_max) * tmax_scale)


def temperature(t, t_shift):
    """
    Temperature function
    ____________________
    Parameters
    ----------
    t : double
        time
    t_shift : double
        initial temperature


    Returns
    -------
    temperature : double
        temperature = t + (t_shift)
    """
    return t + t_shift


def temperature_derivative(_):
    """
    Temperature derivative
    ______________________
    _______
    Returns
    _______
    temperature_derivative : double
        temperature_derivative = 1
    """
    return 1


options = {
    "direct_tmax": True,
    "learning_rate": 1e-5,
    "res_loss_weight": 1e+4,
    "tbeg_loss_weight": 1e+2,
    "ref_loss_weight": 1e+0,
    "mpeak_loss_weight": 0e-1,
    "iter_num": 300000,
    "decay": ("step", 10000, 0.9),
    "e_var_init": 1e+5,
    "e_scale": 5,
    "tmax_scale": 50,
    "k_scale": 1e+1,
    "t_shift": 1400,
    "normalized": True,
    "normalized_input": False,
    "reduced_points": False,
    "last_activation": torch.nn.functional.softplus,
    "melting_temp": 1500,
    "ode_loss_enhancer_power": 0,
    "random_seed": config.random_seed,
}

if not options["direct_tmax"]:
    options["get_k"] = partial(get_k_direct, k_scale=options["k_scale"])
    options["get_k"].__doc__ = get_k_direct.__doc__
else:
    options["get_k"] = get_k_indirect

options["get_e"] = partial(get_e, e_scale=options["e_scale"])
options["get_e"].__doc__ = get_e.__doc__
options["get_t_max"] = partial(get_t_max, tmax_scale=options["tmax_scale"])
options["get_t_max"].__doc__ = get_t_max.__doc__
options["temperature"] = partial(temperature, t_shift=options["t_shift"])
options["temperature"].__doc__ = temperature.__doc__
options["temperature_derivative"] = temperature_derivative
