from functools import partial
import numpy as np
from scipy.integrate import quad


def temperature(t, t_shift):
    """Temperature function"""
    return t + t_shift


def temperature_derivative(_):
    """Temperature derivative"""
    return 1


def func(t, k_param, e_param, temp_func):
    """f(t) = K - E/T(t)"""
    return k_param - e_param / temp_func(t)


def func_prime(t, e_param, temp_func, temp_prime):
    """f'(t) = (E * T'(t)) / (T(t))^2"""
    return (e_param * temp_prime(t)) / (temp_func(t) ** 2)


def create_oxide_function(k_param, e_param, v0, t_0, t_shift=1200):

    temp_func = partial(temperature, t_shift=t_shift)
    f = partial(func, k_param=k_param, e_param=e_param, temp_func=temp_func)

    def function_for_integration(tau):
        return np.exp(f(tau))

    def oxide_function(t):
        """V(t) = V_0 e^(f(t) - f(t_0)) * e^(-integral(t_0, t)(e^f(tau)d tau))"""
        result = np.zeros(t.shape)
        for i in range(len(t)):
            integral = quad(function_for_integration, t_0, t[i])[0]
            result[i] = np.exp(-integral)
        return v0 * np.exp(f(t) - f(t_0)) * result

    return oxide_function
