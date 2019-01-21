"""
Compare the parsing results for the
two different unito CRYSTAL parsers:
    * ejplugins by Chris Sewell
    * pycrystal by Evgeny Blokhin
"""
from __future__ import division
import os
import sys
import random
import logging
import math
import time
from datetime import timedelta

import numpy as np
from ase import Atoms
from ase.geometry import distance
from ase.ga.standard_comparators import InteratomicDistanceComparator
from ase.calculators.calculator import Calculator

from ejplugins.crystal import CrystalOutputPlugin
from pycrystal import CRYSTOUT, CRYSTOUT_Error


logging.getLogger('ejplugins.crystal').setLevel(logging.ERROR)

Ejpcry = CrystalOutputPlugin()
allowed_dt = timedelta(seconds=30)
cmp_atoms = InteratomicDistanceComparator(mic=True)
starttime = time.time()


class Mock(Calculator):
    def __init__(self, *args, **kwargs):
        Calculator.__init__(self)

    def get_property(self, *args, **kwargs):
        return 42


for root, dirs, files in os.walk(sys.argv[1]):
    # NB beware of the broken links (clean e.g. find . -type l -exec rm -f {} \;)

    for filename in files:
        target = root + os.sep + filename
        if not CRYSTOUT.acceptable(target):
            continue

        logging.info("*"*25 + root + os.sep + filename + "*"*25)
        skipped = False

        # Parsing with ejp
        tic = time.time()
        with open(target) as fp:
            try:
                ejp_result = Ejpcry.read_file(fp, log_warnings=False)
            except Exception as err:
                logging.error("EJP FAILED TO PARSE: %s" % str(err))
                skipped = True
        ejp_perf = round(time.time() - tic, 3)

        # Parsing with pcy
        tic = time.time()
        try:
            pcy_result = CRYSTOUT(target)
        except CRYSTOUT_Error as err:
            logging.error("PCY FAILED TO PARSE: %s" % str(err))
            skipped = True
        pcy_perf = round(time.time() - tic, 3)

        if skipped:
            continue

        logging.info("PERFORMANCE: %1.1f" % (ejp_perf / pcy_perf))

        if not pcy_result['prog']:
            logging.error("NOT A KNOWN CODE VERSION")
            continue

        # NB conversion accuracy issues (codata vs. ASE)
        if ejp_result['final'].get('energy') and pcy_result['energy'] is not None:
            if abs(ejp_result['final']['energy']['total_corrected']['magnitude'] -
                   pcy_result['energy']) > 0.0001: # eV
                logging.critical("TOTAL ENERGIES: %s VS. %s" % (
                    ejp_result['final']['energy']['total_corrected']['magnitude'],
                    pcy_result['energy']
                ))

        elif pcy_result['energy'] is None:
            logging.critical("PCY MISSED ENERGY")
            continue
        else:
            logging.error("EJP MISSED ENERGY")
            continue

        # Comparison by SCF
        if abs(len(ejp_result['initial']['scf']) - len(pcy_result['convergence'])) < 2:
            tstep = random.randint(0, len(ejp_result['initial']['scf']) - 1)
            de = ejp_result['initial']['scf'][tstep - 1]['energy']['total']['magnitude'] - \
                ejp_result['initial']['scf'][tstep]['energy']['total']['magnitude']
            if de != 0 and not math.isnan(de):
                power = int(math.floor(math.log(abs(de), 10)))
                if abs(power - pcy_result['convergence'][tstep]) > 1:
                    logging.critical("TOTAL DE IN CONV: %s VS. %s" % (
                        power,
                        pcy_result['convergence'][-1]
                    ))
        else:
            logging.critical("NUMBERS OF THE CONV STEPS DIFFER: %s VS. %s" % (
                len(ejp_result['initial']['scf']),
                len(pcy_result['convergence'])
            ))

        # Comparison by running time
        pcy_d = timedelta(hours=float(pcy_result['duration'] or 0))
        dsplit = map(int, ejp_result['meta']['elapsed_time'].split(':'))
        ejp_d = timedelta(hours=dsplit[0], minutes=dsplit[1], seconds=dsplit[2])
        if abs(pcy_d - ejp_d) > allowed_dt:
            logging.critical("TIME: %s VS. %s" % (pcy_d, ejp_d))

        # NB alpha+beta require ECP subtracting, so we omit
        magmoms = pcy_result['structures'][-1].get_initial_magnetic_moments()
        if magmoms.any():
            if magmoms.tolist() != ejp_result['mulliken']['alpha-beta_electrons']['charges']:
                logging.critical("NON-EQUAL MAGMOMS")

        # Comparison by opt steps
        if ejp_result['optimisation'] and pcy_result['optgeom']:
            if abs(len(ejp_result['optimisation']) - len(pcy_result['optgeom'])) < 2:
                tstep = random.randint(0, min([
                    len(ejp_result['optimisation']),
                    len(pcy_result['optgeom'])
                ]) - 1)
                if abs(ejp_result['optimisation'][tstep]['energy']['total_corrected']['magnitude'] -
                       pcy_result['optgeom'][tstep][-1]) > 0.0001: # eV
                    logging.critical("TOTAL ENERGIES IN OPT: %s VS. %s" % (
                        ejp_result['optimisation'][tstep]['energy']['total_corrected']['magnitude'],
                        pcy_result['optgeom'][tstep][-1]
                    ))
            else:
                logging.critical("NUMBERS OF THE OPT STEPS DIFFER: %s VS. %s" % (
                    len(ejp_result['optimisation']),
                    len(pcy_result['optgeom'])
                ))

        # Structure comparison
        if ejp_result['final']['primitive_cell'].get('cell_vectors') and \
        ejp_result['final']['primitive_cell'].get('ccoords'):
            try:
                ejp_struct = Atoms(
                    symbols=[
                        ''.join([c for c in el if not c.isdigit()]).capitalize()
                        for el in ejp_result['final']['primitive_cell']['symbols']
                    ], # NB atomic_numbers may contain ECPs
                    cell=[
                        ejp_result['final']['primitive_cell']['cell_vectors']['a']['magnitude'],
                        ejp_result['final']['primitive_cell']['cell_vectors']['b']['magnitude'],
                        ejp_result['final']['primitive_cell']['cell_vectors']['c']['magnitude']
                    ],
                    positions=ejp_result['final']['primitive_cell']['ccoords']['magnitude'],
                    pbc=ejp_result['final']['primitive_cell']['pbc']
                )
            except KeyError:
                logging.critical("PROBLEMATIC STRUCTURE: %s" %
                                 ejp_result['final']['primitive_cell']['symbols'])
                continue

            if not all(ejp_struct.get_pbc()): # account non-periodic directions
                adjust, cellpar = False, ejp_struct.get_cell_lengths_and_angles()
                for n in range(3):
                    if cellpar[n] > CRYSTOUT.PERIODIC_LIMIT:
                        adjust, cellpar[n] = True, CRYSTOUT.PERIODIC_LIMIT
                if adjust:
                    ejp_struct.set_cell(cellpar)
        else:
            logging.critical("CANNOT GET EJP STRUCTURE")
            continue

        pcy_struct = pcy_result['structures'][-1]

        args = np.argsort(ejp_struct.positions[:, 2])
        ejp_struct = ejp_struct[args]
        args = np.argsort(pcy_struct.positions[:, 2])
        pcy_struct = pcy_struct[args]

        if len(ejp_struct) != len(pcy_struct):
            logging.critical("N_ATOMS: %s VS. %s" % (len(ejp_struct), len(pcy_struct)))
            continue

        ejp_symbs, pcy_symbs = set(ejp_struct.get_chemical_symbols()), set(pcy_struct.get_chemical_symbols())
        if ejp_symbs != pcy_symbs:
            logging.info("DIFFERENT SYMBOLS: %s VS. %s" % (ejp_symbs, pcy_symbs))
            continue

        ejp_struct.calc, pcy_struct.calc = Mock(), Mock() # code below does not work without this
        if cmp_atoms.looks_like(ejp_struct, pcy_struct):
            dd = distance(pcy_struct, ejp_struct)
            if dd > 2:
                logging.info("STRUCTURE DIFF IS VERY BIG: %s" % dd)
        else:
            logging.critical("STRUCTURES ARE DIFFERENT")

logging.info("Done in %1.2f sc" % (time.time() - starttime))
