# ---------------------------------------------
#
# The file test_output_parser.py is part of the  pycrystal project.  
# Copyright (c) 2019 Andrey Sobolev  
# MIT License
# See http://github.com/tilde-lab/pycrystal for details
#
# ---------------------------------------------


import os
from ase.units import Ha

from pprint import pprint
from pycrystal import CRYSTOUT
from pycrystal.tests import TEST_DIR

DATA_DIR = os.path.join(TEST_DIR, 'data')


def test_08():
    """Single-point calculation; HF; standard AE basis"""
    test_file = os.path.join(DATA_DIR, 'test08.out')
    parser = CRYSTOUT(test_file)
    info = parser.info
    assert info['prog'] == '17 1.0.1'                   # CRYSTAL version
    assert info['finished'] == 2                        # finished without errors
    assert info['energy'] == -5.7132081224317E+02 * Ha  # energy in eV
    assert info['k'] == '8x8x8'                         # Monkhorst-Pack mesh
    assert info['H'] == 'Hartree-Fock'
    assert info['ncycles'][0] == 6
    assert not info['electrons']['basis_set']['ecp']
    assert info['electrons']['basis_set']['bs']['Si']


def test_39():
    """Single-point calculation; DFT; ECP basis"""
    test_file = os.path.join(DATA_DIR, 'test39_dft.out')
    parser = CRYSTOUT(test_file)
    info = parser.info
    assert info['finished'] == 2                        # finished without errors
    assert info['energy'] == -4.8538264773648E+02 * Ha  # energy in eV
    assert info['k'] == '6x6x6'                         # Monkhorst-Pack net
    assert info['H'] == "LDA/PZ_LDA"
    assert info['ncycles'][0] == 9
    assert info['electrons']['basis_set']['ecp']['Ge'][0][1] == (0.82751, -1.26859, -1)
    assert info['electrons']['basis_set']['bs']['Ge'][0][1] == (1.834, 0.4939, 0.006414)


def test_incomplete():
    """Incomplete OUT file, single-point calculation"""
    test_file = os.path.join(DATA_DIR, 'incomplete.out')
    parser = CRYSTOUT(test_file)
    info = parser.info
    assert info['finished'] == 1                       # not finished
    assert info['energy'] is None                      # energy in eV
    assert info['k'] == '12x12x12'                     # Monkhorst-Pack net
    assert not info['ncycles']


def test_optgeom():
    """Geometry optimization"""
    test_file = os.path.join(DATA_DIR, 'optgeom.out')
    parser = CRYSTOUT(test_file)
    info = parser.info
    assert info['prog'] == '14 1.0.2'                  # CRYSTAL version
    assert info['finished'] == 2                       # finished without errors
    assert info['energy'] == -1.400469343370E+03 * Ha  # energy in eV
    assert info['k'] == '3x3x3'                        # Monkhorst-Pack net
    assert len(info['ncycles']) == 11                  # number of optimization steps
    assert len(info['structures']) == 12               # structures before and after each opt step


def test_freqcalc():
    """Phonon dispersion"""
    test_file = os.path.join(DATA_DIR, 'qua_hf_2d_f.out')
    parser = CRYSTOUT(test_file)
    info = parser.info
    assert info['prog'] == '06 1.0'                         # CRYSTAL version
    assert info['finished'] == 2                            # finished without errors
    assert info['energy'] == -1.3167028008915E+03 * Ha      # energy in eV
    assert info['k'] == '3x3x3'                             # Monkhorst-Pack net
    assert info['phonons']['td']['et'] == [0.005306548296]  # Et in Ha/cell
