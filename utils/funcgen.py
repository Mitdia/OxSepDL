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


def generate_random_solution(oxides, amplitudes):
    t_grid = np.linspace(1400, 2000, 451, dtype="float64")
    reference = np.zeros_like(t_grid, dtype="float64")
    amplitude_modifier = np.random.normal(1, 0.05)
    peaks = []
    for oxide, amplitude in zip(oxides.values(), amplitudes):
        e_var = np.random.normal(8e+4, 5e+3)
        t_max_var = max(oxide["Tm"] + np.random.normal(0, 5), 1401)
        k_var = e_var / t_max_var + np.log(e_var / t_max_var ** 2)
        oxide_function = create_oxide_function(float(k_var), e_var, amplitude * amplitude_modifier, t_max_var - 1400,
                                               t_shift=1400)
        data = oxide_function(t_grid - 1400)
        reference += data
        peaks.append(data)
    noise = np.random.normal(0, 1e-2, t_grid.shape)
    kernel = np.ones(30)
    smoothed_noise = np.convolve(noise, kernel, "same")
    smoothed_noise = np.convolve(smoothed_noise, kernel, "same")
    reference += smoothed_noise

    reference[reference < 0] = 0
    return reference, t_grid, peaks
