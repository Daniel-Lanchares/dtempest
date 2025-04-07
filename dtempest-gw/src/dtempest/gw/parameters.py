"""
Parameter space relations for parameter conversion
"""

import numpy as np

# kwargs absorbs all unused arguments.


def mass_1(mass_1_source, redshift, **kwargs):
    return mass_1_source * (1 + redshift)


def mass_2(mass_2_source, redshift, **kwargs):
    return mass_2_source * (1 + redshift)


def chirp_mass(mass_1, mass_2, **kwargs):
    """
    Conversion function: Given some base parameters returns chirp mass.
    """
    return pow(mass_1 * mass_2, 3 / 5) / pow(mass_1 + mass_2, 1 / 5)


def mass_ratio(mass_1, mass_2, **kwargs):
    """
    Conversion function: Given some base parameters returns mass ratio.
    """
    return np.minimum(mass_1, mass_2) / np.maximum(mass_1, mass_2)


def symmetric_mass_ratio(mass_1, mass_2, **kwargs):
    """
    Conversion function: Given some base parameters returns symmetric mass ratio.
    """
    q = mass_ratio(mass_1, mass_2)
    return q / (1 + q ** 2)


def chi_1(a_1, tilt_1=0, **kwargs):
    """
    Conversion function: Given some base parameters returns parallel component of
    unit-less spin.

    If tilt_1 isn't provided assumes aligned spin (might introduce anti-align
    possibility with 50/50 chance of 0 or np.pi).
    """
    return a_1 * np.cos(tilt_1)


def chi_2(a_2, tilt_2=0, **kwargs):
    """
    Conversion function: Given some base parameters returns parallel component of
    unit-less spin.

    If tilt_2 isn't provided assumes aligned spin (might introduce anti-align
    possibility with 50/50 chance of 0 or np.pi).
    """
    return a_2 * np.cos(tilt_2)


def chi_eff(mass_1, mass_2, a_1, a_2, tilt_1=0, tilt_2=0, **kwargs):
    """
    Conversion function: Given some base parameters returns effective spin.
    """
    chi1 = chi_1(a_1, tilt_1)
    chi2 = chi_2(a_2, tilt_2)
    return (mass_1 * chi1 + mass_2 * chi2) / (mass_1 + mass_2)


def chi_1_in_plane(a_1, tilt_1=0, **kwargs):
    """
    Conversion function: Given some base parameters returns the perpendicular
    component of unit-less spin.
    """
    return np.abs(a_1 * np.sin(tilt_1))


def chi_2_in_plane(a_2, tilt_2=0, **kwargs):
    """
    Conversion function: Given some base parameters returns the perpendicular
    component of unit-less spin.
    """
    return np.abs(a_2 * np.sin(tilt_2))


def chi_p(mass_1, mass_2, a_1, a_2, tilt_1=0, tilt_2=0, **kwargs):
    """
    Conversion function: Given some base parameters returns precession spin.
    """
    q = mass_ratio(mass_1, mass_2)
    chi1_p = chi_1_in_plane(a_1, tilt_1)
    chi2_p = chi_2_in_plane(a_2, tilt_2)
    return np.maximum(chi1_p, q * (3 * q + 4) / (4 * q + 3) * chi2_p)


def luminosity_distance(z, **kwargs):
    # import astropy.cosmology as cosmo
    raise NotImplementedError


d_L = luminosity_distance

redef_dict = {  # MANY MISSING (redshift) #TODO
    'mass_1': mass_1,
    'mass_2': mass_2,
    'chirp_mass': chirp_mass,
    'mass_ratio': mass_ratio,
    'symmetric_mass_ratio': symmetric_mass_ratio,
    'chi_eff': chi_eff,
    'chi_p': chi_p,
    'chi_1': chi_1,
    'chi_2': chi_2,
    'chi_1_in_plane': chi_1_in_plane,
    'chi_2_in_plane': chi_2_in_plane,
    'd_L': luminosity_distance,
    'luminosity_distance': luminosity_distance
}

unit_dict = {  # MANY MISSING #TODO
    'mass_1': r'$M_{\odot}$',
    'mass_2': r'$M_{\odot}$',
    'chirp_mass': r'$M_{\odot}$',
    'mass_ratio': r'$ø$',
    'symetric_mass_ratio': r'$ø$',
    'NAP': r'$ø$',
    'chi_eff': r'$ø$',
    'chi_p': r'$ø$',
    'chi_1': r'$ø$',
    'chi_2': r'$ø$',
    'chi_1_in_plane': r'$ø$',
    'chi_2_in_plane': r'$ø$',
    'd_L': r'$Mpc$',
    'luminosity_distance': r'$Mpc$',
    'ra': r'$rad$',
    'dec': r'$rad$'
}

alias_dict = {  # MANY MISSING #TODO
    'mass_1': r'$m_1$',
    'mass_2': r'$m_2$',
    'chirp_mass': r'$\mathcal{M}$',
    'mass_ratio': r'$q$',
    'symetric_mass_ratio': r'$\eta$',
    'NAP': 'Network Antenna Pattern',
    'chi_eff': r'$\chi_{eff}$',
    'chi_p': r'$\chi_{p}$',
    'd_L': r'$d_L$',
    'luminosity_distance': r'$d_L$',
    'ra': r'$ra$',
    'dec': r'$dec$'
}
