from typing import Union

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['figure.figsize'] = [16, 9]
plt.rcParams['font.size'] = 18


def phase_position(time_array: np.array, period_value: float) -> np.array:
    '''
    Расчет положения времени регистрации каждого фотона на
    фазовой кривой для заданного периода.
    '''
    return (time_array % period_value) / period_value


def expected(time_array: np.array, num_bins: float = 20) -> float:
    '''
    Расчет ожидаемого значения в предположении отсутствия периода.
    '''
    return len(time_array)/num_bins


def chi_square(time_array: np.array,
               period: float,
               num_bins: int = 20) -> np.array:
    '''
    Расчет значения хи-квадрат для заданного периода.
    '''
    phase_position_array = phase_position(time_array, period)
    histogram_arr = np.histogram(phase_position_array, bins=num_bins)[0]
    expected_value = expected(time_array, num_bins)
    chi2 = np.sum((histogram_arr - expected_value) ** 2/expected_value ** 2)

    return chi2


def search(times: np.array,
           period_min: float,
           period_max: float,
           num_bins: int = 20,
           num_periods: int = 100) -> Union[np.array, np.array]:
    '''
    Функция для поиска периода методом epoch_folding в заданном
    интервале значений с заданным шагом.
    '''
    periods = np.linspace(period_min, period_max, num_periods)
    stats = np.zeros(num_periods)

    for i, period in enumerate(periods):
        stats[i] = chi_square(times, period, num_bins)
    return periods, stats
