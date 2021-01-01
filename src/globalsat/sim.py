"""
Globalsat simulation model.

Developed by Bonface Osaro and Ed Oughton.

December 2020

"""
import math
import numpy as np
from itertools import tee
from collections import OrderedDict

def system_capacity(constellation, number_of_satellites, params, lut):
    """
    Find the system capacity.

    Parameters
    ----------
    constellation : string
        Consetellation selected for assessment.
    number_of_satellites : int
        Number of satellites in the contellation being simulated.
    params : dict
        Contains all simulation parameters.
    lut : list of tuples
        Lookup table for CNR to spectral efficiency.

    Returns
    -------
    results : list of dicts
        System capacity results generated by the simulation.

    """
    results = []

    distance, satelite_coverage_area = calc_geographic_metrics(number_of_satellites, params)

    random_variations = generate_log_normal_dist_value(
            params['dl_frequency'],
            params['mu'],
            params['sigma'],
            params['seed_value'],
            params['iterations']
        )

    for i in range(0, params['iterations']):

        path_loss = calc_free_space_path_loss(distance, params, i, random_variations)

        antenna_gain = calc_antenna_gain(
            params['speed_of_light'],
            params['antenna_diameter'],
            params['dl_frequency'],
            params['antenna_efficiency']
        )

        eirp = calc_eirp(params['power'], antenna_gain, params['losses'])

        received_power = calc_received_power(eirp, path_loss)

        noise = calc_noise()

        cnr = calc_cnr(received_power, noise)

        spectral_efficiency = calc_spectral_efficiency(cnr, lut)

        capacity = calc_capacity(spectral_efficiency, params['dl_bandwidth'])

        results.append({
            'constellation': constellation,
            'number_of_satellites': number_of_satellites,
            'distance': distance,
            'satelite_coverage_area': satelite_coverage_area,
            'iteration': i,
            'path_loss': path_loss,
            'antenna_gain': antenna_gain,
            'eirp': eirp,
            'received_power': received_power,
            'noise': noise,
            'cnr': cnr,
            'spectral_efficiency': spectral_efficiency,
            'capacity': capacity,
            'capacity_kmsq': capacity / satelite_coverage_area,
        })

    return results


def calc_geographic_metrics(number_of_satellites, params):
    """
    Calculate geographic metrics, including (i) the distance between the transmitter
    and reciever, and (ii) the coverage area for each satellite.

    Parameters
    ----------
    number_of_satellites : int
        Number of satellites in the contellation being simulated.
    params : dict
        Contains all simulation parameters.

    Returns
    -------
    distance : float
        The distance between the transmitter and reciever in km.
    satelite_coverage_area : float
        The area which each satellite covers on Earth's surface in km.

    """
    area_of_earth_covered = (
        params['total_area_earth_km_sq'] *
        params['portion_of_earth_covered']
    )

    network_density = number_of_satellites / area_of_earth_covered

    satelite_coverage_area = (area_of_earth_covered / number_of_satellites)

    mean_distance_between_assets = math.sqrt((1 / network_density)) / 2

    distance = math.sqrt(((mean_distance_between_assets)**2) + ((params['altitude_km'])**2))

    return distance, satelite_coverage_area


def calc_free_space_path_loss(distance, params, i, random_variations):
    """
    Calculate the free space path loss in decibels.

    FSPL(dB) = 20log(d) + 20log(f) + 32.44

    Where distance (d) is in km and frequency (f) is MHz.

    Parameters
    ----------
    distance : float
        Distance between transmitter and receiver in metres.
    params : dict
        Contains all simulation parameters.
    i : int
        Iteration number.
    random_variation : list
        List of random variation components.

    Returns
    -------
    path_loss : float
        The free space path loss over the given distance.

    """
    frequency_MHz = params['dl_frequency'] / 1e6

    path_loss = 20*math.log10(distance) + 20*math.log10(frequency_MHz) + 32.44

    random_variation = random_variations[i]

    return path_loss + random_variation


def generate_log_normal_dist_value(frequency, mu, sigma, seed_value, draws):
    """
    Generates random values using a lognormal distribution, given a specific mean (mu)
    and standard deviation (sigma).

    Original function in pysim5G/path_loss.py.

    The parameters mu and sigma in np.random.lognormal are not the mean and STD of the
    lognormal distribution. They are the mean and STD of the underlying normal distribution.

    Parameters
    ----------
    frequency : float
        Carrier frequency value in Hertz.
    mu : int
        Mean of the desired distribution.
    sigma : int
        Standard deviation of the desired distribution.
    seed_value : int
        Starting point for pseudo-random number generator.
    draws : int
        Number of required values.

    Returns
    -------
    random_variation : float
        Mean of the random variation over the specified itations.

    """
    if seed_value == None:
        pass
    else:
        frequency_seed_value = seed_value * frequency * 100
        np.random.seed(int(str(frequency_seed_value)[:2]))

    normal_std = np.sqrt(np.log10(1 + (sigma/mu)**2))
    normal_mean = np.log10(mu) - normal_std**2 / 2

    random_variation  = np.random.lognormal(normal_mean, normal_std, draws)

    return random_variation


def calc_antenna_gain(c, d, f, n):
    """
    Calculates the antenna gain.

    Parameters
    ----------
    c : float
        Speed of light in meters per second (m/s).
    d : float
        Antenna diameter in meters.
    f : int
        Carrier frequency in Hertz.
    n : float
        Antenna efficiency.

    Returns
    -------
    antenna_gain : float
        Antenna gain in dB.

    """
    #Define signal wavelength
    lambda_wavelength = c / f

    #Calculate antenna_gain
    antenna_gain = 10 * (math.log10(n*((np.pi*d) / lambda_wavelength)**2))

    return antenna_gain


def calc_eirp(power, antenna_gain, losses):
    """
    Calculate the Equivalent Isotropically Radiated Power.

    Equivalent Isotropically Radiated Power (EIRP) = (
        Power + Gain - Losses
    )

    Parameters
    ----------
    power : float
        Transmitter power in watts.
    antenna_gain : float
        Antenna gain in dB.
    losses : float
        Antenna losses in dB.

    Returns
    -------
    eirp : float
        eirp in dB.

    """
    eirp = power + antenna_gain - losses

    return eirp


def calc_received_power(eirp, path_loss):
    """
    Calculates the power received at the User Equipment (UE).

    Parameters
    ----------
    eirp : float
        The Equivalent Isotropically Radiated Power in dB.
    path_loss : float
        The free space path loss over the given distance.

    Returns
    -------
    received_power : float
        The received power at the receiver in dB.

    """
    receiver_gain = 4 # dummy values
    receiver_loss = 4 # dummy values

    received_power = eirp - path_loss + receiver_gain - receiver_loss

    return received_power


def calc_noise():
    """
    Estimates the potential noise.

    Terminal noise can be calculated as:

    “K (Boltzmann constant) x T (290K) x bandwidth”.

    The bandwidth depends on bit rate, which defines the number
    of resource blocks. We assume 50 resource blocks, equal 9 MHz,
    transmission for 1 Mbps downlink.

    Required SNR (dB)
    Detection bandwidth (BW) (Hz)
    k = Boltzmann constant
    T = Temperature (Kelvins) (290 Kelvin = ~16 degrees celcius)
    NF = Receiver noise figure (dB)

    NoiseFloor (dBm) = 10log10(k * T * 1000) + NF + 10log10BW

    NoiseFloor (dBm) = (
        10log10(1.38 x 10e-23 * 290 * 1x10e3) + 1.5 + 10log10(10 x 10e6)
    )

    Parameters
    ----------
    bandwidth : int
        The bandwidth of the carrier frequency (MHz).

    Returns
    -------
    noise : float
        Received noise at the UE receiver in dB.

    """
    k = 1.38e-23 #Boltzmann's constant k = 1.38×10−23 joules per kelvin
    t = 290 #Temperature of the receiver system T0 in kelvins
    b = 0.25 #Detection bandwidth (BW) in Hz

    noise = (10*(math.log10((k*t*1000)))) + (10*(math.log10(b*10**9)))

    return noise


def calc_cnr(received_power, noise):
    """
    Calculate the Carrier-to-Noise Ratio (CNR).

    Returns
    -------
    received_power : float
        The received signal power at the receiver in dB.
    noise : float
        Received noise at the UE receiver in dB.

    Returns
    -------
    cnr : float
        Carrier-to-Noise Ratio (CNR) in dB.

    """
    cnr = received_power - noise

    return cnr


def calc_spectral_efficiency(cnr, lut):
    """
    Given a cnr, find the spectral efficiency.

    Parameters
    ----------
    cnr : float
        Carrier-to-Noise Ratio (CNR) in dB.
    lut : list of tuples
        Lookup table for CNR to spectral efficiency.

    Returns
    -------
    spectral_efficiency : float
        The number of bits per Hertz able to be transmitted.

    """
    spectral_efficiency = 0.1

    for lower, upper in pairwise(lut):

        lower_cnr = lower[0]
        upper_cnr = upper[0]

        if cnr >= lower_cnr and cnr < upper_cnr:
            spectral_efficiency = lower[1]
            return spectral_efficiency

        highest_value = lut[-1]

        if cnr >= highest_value[0]:
            spectral_efficiency = highest_value[1]
            return spectral_efficiency

        lowest_value = lut[0]

        if cnr < lowest_value[1]:
            spectral_efficiency = lowest_value[1]
            return spectral_efficiency

    return spectral_efficiency


def calc_capacity(spectral_efficiency, dl_bandwidth):
    """
    Calculate the channel capacity.

    Parameters
    ----------
    spectral_efficiency : float
        The number of bits per Hertz able to be transmitted.
    dl_bandwidth: float
        The channel bandwidth in Hetz.

    Returns
    -------
    capacity : float
        The channel capacity in Mbps.

    """
    capacity = spectral_efficiency * dl_bandwidth / (10**6)

    return capacity


def pairwise(iterable):
    """
    Return iterable of 2-tuples in a sliding window.

    Parameters
    ----------
    iterable: list
        Sliding window

    Returns
    -------
    list of tuple
        Iterable of 2-tuples

    Example
    -------
    >>> list(pairwise([1,2,3,4]))
        [(1,2),(2,3),(3,4)]

    """
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)
