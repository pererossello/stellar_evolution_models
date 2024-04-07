import os

import numpy as np
from scipy.interpolate import interp1d
import colour
from astropy.io.ascii import read
from astropy import units as u
from astropy.constants import sigma_sb, b_wien

R_sun = 6.957e8 * u.m
L_sun = 3.828e26 * u.W
M_sun = 1.989e30 * u.kg


def read_data(fold):

    files = os.listdir(fold)
    paths = [os.path.join(fold, basename) for basename in files]
    filenames = [file.split(".")[0] for file in files]
    table_dic = {
        filenames[i].split(".")[0]: read(paths[i], data_start=2)
        for i in range(len(paths))
    }

    # We want to sort by mass
    mass_0 = [np.round(tab["mass"][0], 2) for tab in table_dic.values()]
    mass_filename_pairs = [
        (filename.split(".")[0], np.round(tab["mass"][0], 2))
        for filename, tab in table_dic.items()
    ]
    sorted_mass_filename_pairs = sorted(mass_filename_pairs, key=lambda x: x[1])
    table_dic = {
        filename: table_dic[filename] for filename, _ in sorted_mass_filename_pairs
    }
    masses = [mass for _, mass in sorted_mass_filename_pairs]

    for tabname, tab in table_dic.items():
        # And we want the density in SI units
        rhoc_cgs = 10 ** tab["lg(rhoc)"]
        rhoc_SI = rhoc_cgs * (1e-3) * (1e-2) ** (-3)
        tab["lg(rhoc)"] = np.log10(rhoc_SI)

        # We extract radius (in R_sun units) from Steffan-Boltzmann Law
        L = 10 ** (np.array(tab["lg(L)"]) + np.log10(L_sun.value))
        T_eff = 10 ** (tab["lg(Teff)"])
        radius = np.sqrt(L / (4 * np.pi * sigma_sb.value * T_eff**4)) / R_sun
        tab["radius"] = radius

        # Set the time in Myrs
        tab["time"] = tab["time"] * 1e-6

        # get color from wien wavelength
        wien_peaks = b_wien.value / np.array(T_eff)
        tab["wien_peak"] = wien_peaks * 1e9

    return table_dic


def read_element_groups(reg="cen"):
    element_groups = {
        "H": [f"1H_{reg}"],
        "He": [f"4He_{reg}"],
        "C": [f"12C_{reg}", f"13C_{reg}"],
        "N": [f"14N_{reg}"],
        "O": [f"16O_{reg}", f"17O_{reg}", f"18O_{reg}"],
        "Ne": [f"20Ne_{reg}", f"22Ne_{reg}"],
        #'Al': [f'26Al_{reg}']
    }
    return element_groups


def interp_data(tab, M=1000):

    tab_ = {}

    time = tab["time"]
    dt = np.diff(time)
    dt[dt == 0] = np.nan
    min_dt = np.nanmin(dt)

    t_uni = np.arange(time[0], time[-1], min_dt)
    N = len(t_uni)
    s = N // M
    t_uni = t_uni[::s]
    tab_["time"] = t_uni

    vars = list(tab.columns)[1:]

    for var in vars:
        var_ = tab[var]
        interpolate = interp1d(time, var_, kind="linear")
        var_uni = interpolate(t_uni)
        tab_[var] = var_uni

    return tab_


def planck_law(wavelength, temperature):
    """
    Calculate the spectral radiance of a blackbody at a given temperature and wavelength.
    Wavelength in meters, Temperature in Kelvin.
    """
    h = 6.62607015e-34  # Planck constant
    c = 299792458  # Speed of light
    k = 1.380649e-23  # Boltzmann constant

    return (
        (2.0 * h * c**2)
        / (wavelength**5)
        * (1 / (np.exp((h * c) / (wavelength * k * temperature)) - 1))
    )


def get_planck_color(T):

    wavelengths = np.arange(380, 751, 1)  # in nanometers

    spectrum = planck_law(wavelengths * 1e-9, T)
    spectrum /= np.max(spectrum)

    cmfs = colour.colorimetry.MSDS_CMFS["CIE 1931 2 Degree Standard Observer"]

    XYZ = colour.sd_to_XYZ(
        colour.SpectralDistribution(spectrum, wavelengths), cmfs=cmfs
    )
    XYZ_normalized = XYZ / np.max(XYZ)

    RGB = colour.XYZ_to_sRGB(XYZ / 100)

    RGB_normalized = np.clip(RGB, 0, 1)

    return RGB_normalized
