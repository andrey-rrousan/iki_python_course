import warnings

from typing import Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from matplotlib.colors import SymLogNorm as lognorm

warnings.filterwarnings("ignore", category=UserWarning)
from stingray import Lightcurve
from stingray.events import EventList

plt.rcParams['figure.figsize'] = [16, 9]
plt.rcParams['font.size'] = 18


def read_data(filepath: str) -> np.array:
    """
    Load data from txt file

    Args:
        filepath (str): path to data.

    Returns:
        (np.array): array with the data.
    """
    data = np.loadtxt(filepath, skiprows=1)
    with open(filepath, 'r') as file:
        header = ' '.join(file.readlines(1))[:-1]
    print('DATA HEADER')
    print(header)

    return data


def phase_position(time_array: np.array, period_value: float) -> np.array:

    return (time_array % period_value) / period_value


def expected(time_array: np.array, num_bins: float = 20) -> float:

    return len(time_array)/num_bins


def chi_square(time_array: np.array,
               period: float,
               num_bins: int = 20) -> np.array:
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
    periods = np.linspace(period_min, period_max, num_periods)
    stats = np.zeros(num_periods)

    for i, period in enumerate(periods):
        stats[i] = chi_square(times, period, num_bins)
    return periods, stats


def _sinusoid(times: np.array,
              frequency: float,
              baseline: float,
              amplitude: float,
              phase: float) -> np.array:
    return (baseline +
            amplitude * np.sin(2 * np.pi * (frequency * times + phase)))


def periodic_generator(period: float = 50.,
                       obs_length: float = 1000.,
                       mean_countrate: float = 10.,
                       pulsed_fraction: float = 1.0) -> np.array:
    """
    Periodic events generator

    Generates poisson sin-like data

    Args:
        period (float): period of the signal.
        obs_length (float): duration of observation.
        mean_countrate (float): average counts per sec.
        pulsed_fraction (float): from 0 to 1 - sin pulse fraction

    Returns:
        (np.array): generated periodic arrival times

    """
    bin_time = 0.01
    t = np.arange(0, obs_length+bin_time, bin_time)

    # The continuous light curve
    counts = _sinusoid(t, 1 / period, mean_countrate,
                       0.5 * mean_countrate * pulsed_fraction, 0) * bin_time

    lc = Lightcurve(t, counts,
                    gti=[[-bin_time / 2, obs_length + bin_time / 2]],
                    dt=bin_time)

    events = EventList()
    events.simulate_times(lc)
    return events.time


def res_plot(periods: np.array,
             stats: np.array,
             label: str = None,
             line: Union[float, list] = None,
             add_plot: bool = False) -> None:
    """
    Plot the results of epoch folding.

    Args:
    TO BE DONE
    """
    if not add_plot:
        plt.close()
        plt.figure(figsize=(12, 5))
    plt.plot(periods, stats, marker='.', label=label)
    if line is not None:
        plt.vlines(line, stats.min(), stats.max(), colors='tab:red', alpha=0.5)
    # plt.xticks(np.linspace(periods.min(), periods.max(), 1))
    # plt.yscale('log')
    plt.ylabel('Значение хи-квадрат')
    plt.xlabel('Период, с')


def phase_plot(time_array: np.array,
               period_value: float,
               num_bins: int = 20,
               add_plot: bool = False) -> None:
    """
    Function to plot the phase profile of the observation with given period.

    Args:
        time (np.array): time data of the observation.
        period (float): time period to plot the phase profile for.
        bins (int) = 20: number of bins to divide the data into.
    """
    if not add_plot:
        plt.close()
        plt.figure(figsize=(10, 5))
    plt.hist((time_array % period_value) / period_value, bins=num_bins,
             range=(0, 1), histtype='step')
    plt.xlabel('Фаза периода')
    plt.ylabel('Число фотонов')


def folding_plot(data, period):

    cum_arr = np.zeros(data.shape, dtype=bool)
    plt.figure(figsize=(16, 10))
    for i in range(int(30/period)+1):
        plt.subplot(2, 1, 1)
        interval_idx = (data < period*(i+1)) * (data >= period*(i))
        plt.hist(data[interval_idx],
                 bins=np.linspace(0, 30, 91), histtype='stepfilled')

        plt.subplot(2, 1, 2)
        cum_arr = np.logical_or(cum_arr, interval_idx)
        plt.hist(data[cum_arr] % period,
                 bins=np.linspace(0, period, 9), zorder=-i)
    plt.hist(data[cum_arr] % period,
             bins=np.linspace(0, period, 9), histtype='step', edgecolor='black', linewidth=2)

    plt.subplot(2, 1, 1)
    plt.xticks(range(0, 31, 3))
    plt.xlabel('Время, с')  # Зададим название оси x
    plt.ylabel('Число фотонов, шт')  # Зададим название оси y

    plt.subplot(2, 1, 2)
    plt.xlabel('Положение на периоде, с')  # Зададим название оси x
    plt.ylabel('Число фотонов, шт')  # Зададим название оси y


DATA_LINK = (r'.\data\CenX3.fits')


def real_data_read():
    data = fits.getdata(DATA_LINK)
    x = data['X']
    y = data['Y']
    t = data['TIME']

    _filter = np.where((data['GRADE'] >= 0) * (data['GRADE'] <= 26) *
                       (data['PI'] >= 35) * (data['PI'] <= 1935))

    x, y, t = x[_filter], y[_filter], t[_filter]
    xy_filter = np.where((x >= 490) * (x <= 510) *
                         (y >= 490) * (y <= 510))

    t = t[xy_filter]
    t_slice = t[np.logical_and(t < (1.866e8 + 7000.), t > (1.866e8 + 3500.))]

    return t_slice


def real_data_show():
    data = fits.getdata(DATA_LINK)
    hist_data = np.histogram2d(data['X'], data['Y'],
                               bins=1000, range=[[0, 1000], [0, 1000]])[0]

    fig, ax = plt.subplots(figsize=(10, 10))

    ax.imshow(hist_data, norm=lognorm(1))

    rect = mpl.patches.Rectangle((490, 490), 20, 20, linewidth=1.0,
                                 edgecolor='r', facecolor='none',
                                 linestyle='--')
    ax.add_patch(rect)

    plt.xlim(300, 800)
    plt.ylim(250, 750)

    plt.show()
