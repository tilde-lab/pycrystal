
import math
import re
import time
import copy
from fractions import Fraction
from pathlib import PurePath

from numpy import cross
from ase.data import chemical_symbols, atomic_numbers
from ase.geometry import cellpar_to_cell
from ase.units import Hartree
from ase import Atoms


def find_all(a_str, sub):
    """
    String finder iterator
    """
    start = 0
    while True:
        start = a_str.find(sub, start)
        if start == -1:
            return
        yield start
        start += len(sub)


def metric(v):
    """
    Get the direction of a vector
    """
    return [int(math.copysign(1, x)) if x else 0 for x in v]


class CRYSTOUT_Error(Exception):
    def __init__(self, msg, code=0):
        Exception.__init__(self)
        self.msg = msg
        self.code = code

    def __str__(self):
        return repr(self.msg)


# noinspection PyTypeChecker
class CRYSTOUT(object):
    patterns = {
        'Etot': re.compile(r"\n\sTOTAL ENERGY\(.{2,3}\)\(.{2}\)\(.{3,4}\)\s(\S{20})\s{1,10}DE(?!.*\n\s"
                           r"TOTAL ENERGY\(.{2,3}\)\(.{2}\)\(.{3,4}\)\s)", re.DOTALL),
        'dEtot': re.compile(r"\sTOTAL ENERGY\(.{2,3}\)\(.{2}\)\(.{3,4}\).{21}\s{1,10}DE\s*[(AU)]*\s*"
                            r"([-+\d.E]*)", re.DOTALL),
        'pEtot': re.compile(r"\n\sTOTAL ENERGY\s(.+?)\sCONVERGENCE"),
        'syminfos': re.compile(r"SYMMOPS - TRANSLATORS IN FRACTIONA\w{1,2} UNITS(.+?)\n\n", re.DOTALL),
        'frac_primitive_cells': re.compile(r"\n\sPRIMITIVE CELL(.+?)ATOM BELONGING TO THE ASYMMETRIC UNIT", re.DOTALL),
        'molecules': re.compile(r"\n\sATOMS IN THE ASYMMETRIC UNIT(.+?)"
                                r"ATOM BELONGING TO THE ASYMMETRIC UNIT", re.DOTALL),
        'cart_vectors': re.compile(r"DIRECT LATTICE VECTORS CARTESIAN COMPONENTS \(ANGSTROM\)(.+?)\n\n",
                                   re.DOTALL),
        'crystallographic_cell': re.compile(r"\n\sCRYSTALLOGRAPHIC(.+?)\n\sT\s=", re.DOTALL),
        'at_str': re.compile(r"^\s{0,6}\d{1,4}\s"),
        'charges': re.compile(r"ALPHA\+BETA ELECTRONS\n"
                              r"\sMULLIKEN POPULATION ANALYSIS(.+?)OVERLAP POPULATION CONDENSED TO ATOMS",
                              re.DOTALL),
        'magmoms': re.compile(r"ALPHA-BETA ELECTRONS\n"
                              r"\sMULLIKEN POPULATION ANALYSIS(.+?)OVERLAP POPULATION CONDENSED TO ATOMS",
                              re.DOTALL),
        'icharges': re.compile(r"\n\sATOMIC NUMBER(.{4}),\sNUCLEAR CHARGE(.{7}),"),
        'starting': re.compile(r"EEEEEEEEEE STARTING(.+?)\n"),
        'starting_ion': re.compile(r"\n OPTOPTOPTOPTOPT"),
        'scf_converge': re.compile(r" == SCF ENDED - CONVERGENCE ON ENERGY {6}E\(AU\)\s*"
                                   r"([-+\dE.]*)\s*CYCLES\s*([\d]*)", re.DOTALL),
        'ion_converge': re.compile(r" \* OPT END - CONVERGED \* E\(AU\):\s*([-+\dE.]*)\s*POINTS\s*([\d]*) \*",
                                   re.DOTALL),
        'ending': re.compile(r"EEEEEEEEEE TERMINATION(.+?)\n"),
        'conduction_states': re.compile(r"(INSULATING|CONDUCTING) STATE(.*?)TTTTTTT", re.DOTALL),
        'top_valence': re.compile(r"TOP OF VALENCE BANDS - {4}BAND\s*(\d*); K\s*(\d*); EIG\s*([-.E\d]*) AU", re.DOTALL),
        'bottom_virtual': re.compile(r"BOTTOM OF VIRTUAL BANDS - BAND\s*(\d*); K\s*(\d*); EIG\s*([-.E\d]*) AU",
                                     re.DOTALL),
        'band_gap': re.compile(r"(DIRECT|INDIRECT) ENERGY BAND GAP:\s*([.\d]*)", re.DOTALL),
        'e_fermi': re.compile(r"EFERMI\(AU\)\s*([-+.E\d]*)", re.DOTALL),
        'freqs': re.compile(r"DISPERSION K POINT(.+?)FREQ\(CM\*\*-1\)", re.DOTALL),
        'n_atoms': re.compile(r"\sN. OF ATOMS PER CELL\s*(\d*)", re.DOTALL),
        'n_shells': re.compile(r"\sNUMBER OF SHELLS\s*(\d*)", re.DOTALL),
        'n_ao': re.compile(r"\sNUMBER OF AO\s*(\d*)", re.DOTALL),
        'n_electrons': re.compile(r"\sN. OF ELECTRONS PER CELL\s*(\d*)", re.DOTALL),
        'n_core_el': re.compile(r"\sCORE ELECTRONS PER CELL\s*(\d*)", re.DOTALL),
        'n_symops': re.compile(r"\sN. OF SYMMETRY OPERATORS\s*(\d*)", re.DOTALL),
        'gamma_freqs': re.compile(r"\(HARTREE\*\*2\)\s*\(CM\*\*-1\)\s*\(THZ\)\s*\(KM/MOL\)(.+?)"
                                  r"NORMAL MODES NORMALIZED TO CLASSICAL AMPLITUDES", re.DOTALL),
        'ph_eigvecs': re.compile(r"NORMAL MODES NORMALIZED TO CLASSICAL AMPLITUDES(.+?)\*{79}", re.DOTALL),
        'needed_disp': re.compile(r"\d{1,4}\s{2,6}(\d{1,4})\s{1,3}\w{1,2}\s{11,12}(\w{1,2})\s{11,12}\d{1,2}"),
        'symdisps': re.compile(r"N {3}LABEL SYMBOL DISPLACEMENT {5} SYM.(.*)NUMBER OF IRREDUCIBLE ATOMS",
                               re.DOTALL),
        'ph_k_degeneracy': re.compile(r"K {7}WEIGHT {7}COORD(.*)AND RECIPROCAL LATTICE VECTORS", re.DOTALL),
        'supmatrix': re.compile(r"EXPANSION MATRIX OF PRIMITIVE CELL(.+?)\sNUMBER OF ATOMS PER SUPERCELL",
                                re.DOTALL),
        'cyc': re.compile(r"\n\sCYC\s(.+?)\n"),
        'enes': re.compile(r"\n\sTOTAL ENERGY\((.+?)\n"),
        't1': re.compile(r"\n\sMAX\sGRADIENT(.+?)\n"),
        't2': re.compile(r"\n\sRMS\sGRADIENT(.+?)\n"),
        't3': re.compile(r"\n\sMAX\sDISPLAC.(.+?)\n"),
        't4': re.compile(r"\n\sRMS\sDISPLAC.(.+?)\n"),
        'version': re.compile(r"\s\s\s\s\sCRYSTAL\d{2}(.*)\*\n", re.DOTALL),
        'pv': re.compile(r"\n PV {12}:\s(.*)\n"),
        'ts': re.compile(r"\n TS {12}:\s(.*)\n"),
        'et': re.compile(r"\n ET {12}:\s(.*)\n"),
        'T': re.compile(r"\n AT \(T =(.*)K, P =(.*)MPA\):\n"),
        'entropy': re.compile(r"\n ENTROPY {7}:\s(.*)\n"),
        'C': re.compile(r"\n HEAT CAPACITY :\s(.*)\n"),
        'elastic_constants': re.compile(r" SYMMETRIZED ELASTIC CONSTANTS .*\n\n"
                                        r" \| ([.\d\s]*) \|\n"
                                        r" \| ([.\d\s]*) \|\n"
                                        r" \| ([.\d\s]*) \|\n"
                                        r" \| ([.\d\s]*) \|\n"
                                        r" \| ([.\d\s]*) \|\n"
                                        r" \| ([.\d\s]*) \|\n"),
        'elastic_moduli': re.compile(r" ELASTIC MODULI .*\n\n"
                                     r" \| ([-.\d\s]*) \|\n"
                                     r" \| ([-.\d\s]*) \|\n"
                                     r" \| ([-.\d\s]*) \|\n"
                                     r" \| ([-.\d\s]*) \|\n"
                                     r" \| ([-.\d\s]*) \|\n"
                                     r" \| ([-.\d\s]*) \|\n"),
        'effective_moduli': re.compile(r"K_V\s*G_V.*\n\n([-.\d\s]*)"),
    }

    # this is the limiting distance,
    # if more, the direction is considered non-periodic
    # be careful, as this has no physical meaning
    # NB non-periodic component(s) are assigned 500 A in CRYSTAL
    PERIODIC_LIMIT = 50

    def __init__(self, filename, **kwargs):

        self.data = ''              # file contents
        self.pdata = None
        self.properties_calc, self.crystal_calc = False, False

        self.info = {
            'warns': [],
            'prog': None,           # code version
            'techs': [],
            'finished': 0x0,
            'duration': None,
            'input': None,
            'structures': [],       # list of valid ASE objects
            'energy': None,         # in eV
            'e_accuracy': None,
            'scf_conv': None,
            'ion_conv': None,
            'H': None,
            'H_types': [],          # can be 0x1, 0x2, 0x4, and 0x5
            'tol': None,
            'k': None,
            'smear': None,          # in a.u.
            'spin': False,
            'lockstate': None,
            'convergence': [],      # zero-point energy convergence
            'conduction': [],
            'optgeom': [],          # optimization convergence, list of lists, 5 values each
            'ncycles': [],          # number of cycles at each optimisation step
            'n_atoms': None,
            'n_shells': None,
            'n_ao': None,
            'n_electrons': None,
            'n_core_el': None,
            'n_symops': None,

            'electrons': {
                'basis_set': None,  # LCAO Gaussian basis sets in form: {'bs': {...}, 'ecp': {...}}
                'eigvals': {},      # raw eigenvalues {k:{alpha:[...], beta:[...]},}
                'projected': [],    # raw eigenvalues [..., ...] for total DOS smearing
                'dos': {},          # in advance pre-computed DOS
                'bands': {}         # in advance pre-computed band structure
            },
            'phonons': {
                'modes': {},
                'irreps': {},
                'ir_active': {},
                'raman_active': {},
                'ph_eigvecs': {},
                'ph_k_degeneracy': {},
                'dfp_disps': [],
                'dfp_magnitude': None,
                'dielectric_tensor': False,
                'zpe': None,
                'td': None
            },
            'elastic': {
                'elastic_constants': [],
                'elastic_moduli': [],
                'K_V': None,
                'G_V': None,
                'K_R': None,
                'G_R': None,
                'K': None,
                'G': None,
                'E': None,
                'v': None,
                # 'seismic_velocities': {}
            },
        }

        open_close = isinstance(filename, (str, PurePath))
        if open_close:
            raw_data = open(filename).read()
        else:
            raw_data = filename.read()

        # normalize breaks and get rid of the possible odd MPI incusions in important data
        raw_data = raw_data.replace('\r\n', '\n').replace('\r', '\n').replace('FORTRAN STOP\n', '')
        parts_pointer = list(find_all(raw_data, "*                              MAIN AUTHORS"))

        # determine whether to deal with the CRYSTAL and/or PROPERTIES output formats
        if len(parts_pointer) > 1:
            if (not CRYSTOUT.is_properties(raw_data[parts_pointer[1]:]) and
                    len(raw_data[parts_pointer[1]:]) > 2000): # in case of empty properties outputs
                raise CRYSTOUT_Error('File contains several merged outputs, currently not supported!')
            else:
                self.data = raw_data[parts_pointer[0]: parts_pointer[1]]
                self.pdata = raw_data[parts_pointer[1]:]
                self.properties_calc, self.crystal_calc = True, True
        else:
            if not CRYSTOUT.is_properties(raw_data[parts_pointer[0]:]):
                self.data = raw_data[parts_pointer[0]:]
                self.crystal_calc = True
            else:
                self.pdata = raw_data[parts_pointer[0]:]
                self.properties_calc = True

        if not self.crystal_calc and not self.properties_calc:
            raise CRYSTOUT_Error('Though this file looks similar to CRYSTAL output, its format is unknown!')

        if self.crystal_calc:
            self.info['duration'] = self.get_duration()
            self.comment, self.info['input'], self.info['prog'] = self.get_input_and_meta(raw_data[0:parts_pointer[0]])
            self.molecular_case = ' MOLECULAR CALCULATION' in self.data
            self.info['energy'] = self.get_etot()
            self.info['e_accuracy'] = self.get_detot()
            self.info['structures'] = self.get_structures()

            self.decide_charges()
            self.decide_finished()
            self.decide_method()
            self.decide_scfdata()

            self.info['scf_conv'], self.info['ion_conv'] = self.get_convergence()
            self.info['conduction'] = self.get_conduction()

            self.info['electrons']['basis_set'] = self.get_bs()

            self.info['phonons']['ph_k_degeneracy'] = self.get_k_degeneracy()
            self.info['phonons']['modes'], self.info['phonons']['irreps'], self.info['phonons']['ir_active'], \
                self.info['phonons']['raman_active'] = self.get_phonons()
            self.info['phonons']['ph_eigvecs'] = self.get_ph_eigvecs()
            self.info['phonons']['dfp_disps'], self.info['phonons']['dfp_magnitude'] = self.get_ph_sym_disps()
            self.info['phonons']['dielectric_tensor'] = self.get_static_dielectric_tensor()

            # extract zero-point energy, depending on phonons presence
            if self.info['phonons']['modes']:
                self.info['phonons']['zpe'] = self.get_zpe()
                self.info['phonons']['td'] = self.get_td()

            # format ph_k_degeneracy
            if self.info['phonons']['ph_k_degeneracy']:
                bz, d = [], []
                for k, v in self.info['phonons']['ph_k_degeneracy'].items():
                    bz.append(self.info['phonons']['ph_k_degeneracy'][k]['bzpoint'])
                    d.append(self.info['phonons']['ph_k_degeneracy'][k]['degeneracy'])
                self.info['phonons']['ph_k_degeneracy'] = {}
                for i in range(len(bz)):
                    self.info['phonons']['ph_k_degeneracy'][bz[i]] = d[i]

            # get numbers of electrons, ao, etc
            self.info['n_atoms'] = self.get_number('n_atoms')
            self.info['n_shells'] = self.get_number('n_shells')
            self.info['n_ao'] = self.get_number('n_ao')
            self.info['n_electrons'] = self.get_number('n_electrons')
            self.info['n_core_el'] = self.get_number('n_core_el')
            self.info['n_symops'] = self.get_number('n_symops')

            # get elastic constants
            self.info['elastic']['elastic_constants'] = self.get_elastic('elastic_constants')
            self.info['elastic']['elastic_moduli'] = self.get_elastic('elastic_moduli')

            if self.info['elastic']['elastic_moduli'] and ' OPTIMIZE THE STRUCTURE AND RE-RUN\n' in self.data:
                raise CRYSTOUT_Error('Inadequate elastic calculation: additional optimization needed')

            k_v, g_v, k_r, g_r, k, g, e, v = self.get_effective_elastic_moduli()
            self.info['elastic']['K_V'] = k_v
            self.info['elastic']['G_V'] = g_v
            self.info['elastic']['K_R'] = k_r
            self.info['elastic']['G_R'] = g_r
            self.info['elastic']['K'] = k
            self.info['elastic']['G'] = g
            self.info['elastic']['E'] = e
            self.info['elastic']['v'] = v

        if self.properties_calc and not self.crystal_calc:
            raise CRYSTOUT_Error('PROPERTIES output with insufficient information omitted!')

    def warning(self, msg):
        self.info['warns'].append(msg)

    def __repr__(self):
        return repr(self.info)

    def __getitem__(self, key):
        return self.info.get(key)

    @staticmethod
    def detect(test_string):
        if "*                              MAIN AUTHORS" in test_string:
            return True
        return False

    @staticmethod
    def acceptable(filename):
        open_close = isinstance(filename, (str, PurePath))
        if open_close:
            f = open(filename, 'r')
        else:
            f = filename

        counter = 0

        while counter < 700:
            fingerprint = f.readline()
            if CRYSTOUT.detect(fingerprint):
                if open_close:
                    f.close()
                return True
            counter += 1
        if open_close:
            f.close()

        return False

    @staticmethod
    def is_properties(piece_of_data):
        if (" RESTART WITH NEW K POINTS NET" in piece_of_data
                or " CRYSTAL - PROPERTIES" in piece_of_data
                or "Wavefunction file can not be found" in piece_of_data):
            return True
        else:
            return False

    def get_cart2frac(self):
        matrix = []
        vectors = self.patterns['cart_vectors'].findall(self.data)

        if vectors:
            lines = vectors[-1].splitlines()
            for line in lines:
                vector = line.split()
                try:
                    vector[0] and float(vector[0])
                except (ValueError, IndexError):
                    continue
                for i in range(3):
                    vector[i] = float(vector[i])
                    # check whether angstroms are used instead of fractions
                    if vector[i] > self.PERIODIC_LIMIT:
                        vector[i] = self.PERIODIC_LIMIT
                matrix.append(vector)
        else:
            if not self.molecular_case:
                raise CRYSTOUT_Error('Unable to extract cartesian vectors!')

        return matrix

    def get_structures(self):
        structures = []
        if self.molecular_case:
            used_pattern = self.patterns['molecules']
        else:
            used_pattern = self.patterns['frac_primitive_cells']

        strucs = used_pattern.findall(self.data)

        if not strucs:
            raise CRYSTOUT_Error('No structure was found!')

        cell = self.get_cart2frac()

        for crystal_data in strucs:
            symbols, parameters, atompos, pbc = [], [], [], [True, True, True]

            if self.molecular_case:
                pbc = False

            crystal_data = re.sub(' PROCESS(.{32})WORKING\n', '',
                                  crystal_data) # Warning! MPI statuses may spoil valuable data

            # this is to account correct cart->frac atomic coords conversion using cellpar_to_cell ASE routine
            # 3x3 cell is used only here to obtain ab_normal and a_direction
            ab_normal = [0, 0, 1] if self.molecular_case else metric(cross(cell[0], cell[1]))
            a_direction = None if self.molecular_case else metric(cell[0])

            other = self.patterns['crystallographic_cell'].search(crystal_data)
            if other is not None:
                crystal_data = crystal_data.replace(other.group(), "") # delete other cells info except primitive cell

            lines = crystal_data.splitlines()
            for li in range(len(lines)):
                if 'ALPHA      BETA       GAMMA' in lines[li]:
                    parameters = lines[li + 1].split()
                    try:
                        parameters = [float(i) for i in parameters]
                    except ValueError:
                        raise CRYSTOUT_Error('Cell data are invalid: ' + lines[li + 1])

                elif self.patterns['at_str'].search(lines[li]):
                    atom = lines[li].split()
                    if len(atom) in [7, 8] and len(atom[-2]) > 7:
                        for i in range(4, 7):
                            try:
                                atom[i] = round(float(atom[i]), 10)
                            except ValueError:
                                raise CRYSTOUT_Error('Atomic coordinates are invalid!')

                        # NB we lose here the non-equivalency in the same atom types, denoted by different integers
                        # For magmoms refer to the corresponding property
                        atom[3] = ''.join([letter for letter in atom[3] if not letter.isdigit()]).capitalize()
                        if atom[3] == 'Xx':
                            atom[3] = 'X'
                        symbols.append(atom[3])
                        atomdata = atom[4:7]
                        # atomdata.append(atom[1]) # irreducible (T or F)
                        atompos.append(atomdata)

            if len(atompos) == 0:
                raise CRYSTOUT_Error('No atoms found, cell info is corrupted!')

            if parameters and len([x for x in parameters if x > 0.75]) < 6:
                raise CRYSTOUT_Error('Cell is collapsed!') # cell collapses are known in CRYSTAL RESTART outputs

            # check whether angstroms are used instead of fractions
            if pbc:
                for i in range(3):
                    if parameters[i] > self.PERIODIC_LIMIT:
                        parameters[i] = self.PERIODIC_LIMIT
                        pbc[i] = False

                        # TODO: account case with not direct angles?
                        for j in range(len(atompos)):
                            atompos[j][i] /= self.PERIODIC_LIMIT

                matrix = cellpar_to_cell(parameters, ab_normal,
                                         a_direction)
                # TODO: ab_normal, a_direction may in some cases belong to completely other structure!
                structures.append(Atoms(symbols=symbols, cell=matrix, scaled_positions=atompos, pbc=pbc))
            else:
                structures.append(
                    Atoms(symbols=symbols, cell=[self.PERIODIC_LIMIT, self.PERIODIC_LIMIT, self.PERIODIC_LIMIT],
                          positions=atompos, pbc=False))

        return structures

    def get_conduction(self): # FIXME: check for errors in CRYSTAL17 outputs
        result = []
        states = self.patterns['conduction_states'].findall(self.data)
        for state in states:
            state_dict = {'state': state[0]}
            if state[0] == "INSULATING":
                # dealing with band gaps
                try:
                    top = self.patterns['top_valence'].search(self.data).groups()
                except AttributeError:
                    # old CRYSTAL version, may be?
                    continue
                bottom = self.patterns['bottom_virtual'].search(self.data).groups()
                state_dict['top_valence'] = int(top[0])
                state_dict['bottom_virtual'] = int(bottom[0])
                gap_re = self.patterns['band_gap'].search(self.data)
                if gap_re is not None:
                    bg_type, bg = gap_re.groups()
                    state_dict['band_gap'] = float(bg)
                    state_dict['band_gap_type'] = bg_type
                else:
                    # try to deduce band gap from eigenvalues
                    state_dict['band_gap_type'] = "INDIRECT" if top[1] != bottom[1] else "DIRECT"
                    state_dict["band_gap"] = (float(bottom[2]) - float(top[2])) * Hartree
            else:
                # dealing with Fermi energies
                state_dict['e_fermi'] = float(self.patterns['e_fermi'].search(self.data).groups()[0])
                state_dict['e_fermi_units'] = 'Ha'
            result.append(state_dict)
        return result

    def get_etot(self):
        e = self.patterns['Etot'].search(self.data)

        if e is not None:
            return float(e.groups()[0]) * Hartree
        else:
            if '  CENTRAL POINT ' in self.data:
                phonon_e = self.data.split('  CENTRAL POINT ')[-1].split("\n", 1)[0]
                phonon_e = phonon_e.split()[0]
                return float(phonon_e) * Hartree
            else:
                self.warning('No energy found!')
                return None

    def get_number(self, pat_name):
        num = self.patterns[pat_name].search(self.data)

        if num is not None:
            return int(num.groups()[0])
        return None

    def get_detot(self):
        de = self.patterns['dEtot'].search(self.data)
        if de is not None and de.groups()[0]:
            # it might happen that DE is equal to NaN
            return float(de.groups()[0]) * Hartree
        return None

    def get_convergence(self):
        """
        Returns electronic and ionic convergence
        """
        if not self.info['convergence']:
            return None, None
        # electronic convergence
        conv_el_re = self.patterns['scf_converge'].search(self.data)
        conv_el = False if conv_el_re is None else bool(conv_el_re.groups())
        # ionic convergence
        optgeom_re = self.patterns['starting_ion'].search(self.data)
        if optgeom_re is None:
            return conv_el, None
        conv_ion_re = self.patterns['ion_converge'].search(self.data)
        conv_ion = False if conv_ion_re is None else bool(conv_ion_re.groups())
        return conv_el, conv_ion

    def get_phonons(self):
        if not "U   U  EEEE  N   N   CCC  Y   Y" in self.data:
            return None, None, None, None

        freqdata = []
        freqsp = self.patterns['freqs'].findall(self.data)
        if freqsp:
            for i in freqsp:
                freqdata.append([_f for _f in i.strip().splitlines() if _f])
        else:
            freqsp = self.patterns['gamma_freqs'].search(self.data)
            if freqsp is None:
                return None, None, None, None
            else:
                freqdata.append([_f for _f in freqsp.group(1).strip().splitlines() if _f])
        bz_modes, bz_irreps, kpoints = {}, {}, []
        ir_active, raman_active = [], []
        for freqset in freqdata:
            modes, irreps = [], []
            for line in freqset:
                if " R( " in line or " C( " in line: # k-coords
                    coords = line.split("(")[-1].split(")")[0].split()
                    kpoints.append(" ".join(coords))
                    continue

                if "(" in line and ")" in line: # filter lines with freqs: condition 1 from 3
                    val = line.split()
                    if len(val) < 5:
                        continue # filter lines with freqs: condition 2 from 3
                    try:
                        float(val[2]) + float(val[3])
                    except ValueError:
                        continue # filter lines with freqs: condition 3 from 3

                    nmodes = [_f for _f in val[0].split("-") if _f]
                    if len(nmodes) == 1: # fixed place for numericals
                        mplr = int(val[1]) - int(val[0].replace("-", "")) + 1
                        for i in range(mplr):
                            modes.append(float(val[3]))
                            irrep = val[5].replace("(", "").replace(")", "").strip()
                            if not irrep:
                                irrep = val[6].replace("(", "").replace(")", "").strip()
                            irrep = irrep.replace('"', "''")
                            irreps.append(irrep)
                    else: # fixed place for numericals
                        mplr = int(nmodes[1]) - int(nmodes[0]) + 1
                        for i in range(mplr):
                            modes.append(float(val[2]))
                            irrep = val[4].replace("(", "").replace(")", "").strip()
                            if not irrep:
                                irrep = val[5].replace("(", "").replace(")", "").strip()
                            irrep = irrep.replace('"', "''")
                            irreps.append(irrep)
                    # IR / RAMAN data ( * mplr )
                    c = 0
                    for i in range(-4, 0):
                        if val[i] in ['A', 'I']:
                            if c == 0:
                                ir_active.extend([val[i]] * mplr)
                            else:
                                raman_active.extend([val[i]] * mplr)
                            c += 1

            if not kpoints:
                BZ_point_coord = '0 0 0'
            else:
                BZ_point_coord = kpoints[-1]

            # normalize special symmerty point coords, if exist
            if self.info['phonons']['ph_k_degeneracy']:
                BZ_point_coord = self.info['phonons']['ph_k_degeneracy'][BZ_point_coord]['bzpoint']

            bz_modes[BZ_point_coord] = modes
            bz_irreps[BZ_point_coord] = irreps
        return bz_modes, bz_irreps, ir_active, raman_active

    def get_ph_eigvecs(self):
        if not self.info['phonons']['modes']:
            return None

        eigvecdata = []
        eigvecsp = self.patterns['ph_eigvecs'].search(self.data)
        if eigvecsp:
            eigvecsp = eigvecsp.group(1)
            parts = eigvecsp.split("DISPERSION K POINT")
            parts[0] = parts[0].split("LO-TO SPLITTING")[0] # no lo-to splitting account at the moment
            for bzpoint in parts:
                eigvecdata.append(bzpoint.split("FREQ(CM**-1)"))
        else:
            return None

        natseq = list(range(1, len(self.info['structures'][-1]) + 1))
        bz_eigvecs, kpoints = {}, []
        for set in eigvecdata:
            ph_eigvecs = []
            for i in set:
                rawdata = [_f for _f in i.strip().splitlines() if _f]
                freqs_container = []
                involved_atoms = []
                for j in rawdata:
                    if " R( " in j or " C( " in j: # k-coords
                        coords = j[20:-18].strip().split()
                        kpoints.append(" ".join(coords))
                        continue
                    vectordata = j.split()
                    if vectordata[0] == 'AT.':
                        involved_atoms.append(int(vectordata[1]))
                        vectordata = vectordata[4:]
                    elif vectordata[0] == 'Y' or vectordata[0] == 'Z':
                        vectordata = vectordata[1:]
                    else:
                        continue

                    if not len(freqs_container):
                        for _ in range(len(vectordata)):
                            freqs_container.append([]) # 6 (or 3) columns
                    for k in range(len(vectordata)):
                        vectordata[k] = float(vectordata[k])
                        if vectordata[k] != vectordata[k]: # test NaN
                            raise CRYSTOUT_Error('Eigenvector error: NaN occured!')
                        freqs_container[k].append(vectordata[k])

                for fn in range(len(freqs_container)):
                    for at in natseq:
                        if at in involved_atoms:
                            continue
                        # insert fake zero vectors for atoms which are not involved in a vibration
                        for _ in range(3):
                            freqs_container[fn].insert((at - 1) * 3, 0)
                    ph_eigvecs.append(freqs_container[fn])

                if 'ANTI-PHASE' in i:
                    self.warning(
                        'Phase and anti-phase eigenvectors found at k=(%s), the latter will be omitted' % kpoints[-1])
                    break

            if len(ph_eigvecs) != len(self.info['phonons']['modes']['0 0 0']):
                raise CRYSTOUT_Error('Number of eigenvectors does not correspond to the number of freqs!')

            if not kpoints:
                BZ_point_coord = '0 0 0'
            else:
                BZ_point_coord = kpoints[-1]

            # normalize special symmerty point coords, if exist
            if self.info['phonons']['ph_k_degeneracy']:
                BZ_point_coord = self.info['phonons']['ph_k_degeneracy'][BZ_point_coord]['bzpoint']
            bz_eigvecs[BZ_point_coord] = ph_eigvecs

        return bz_eigvecs

    def get_k_degeneracy(self):
        ph_k_degeneracy = self.patterns['ph_k_degeneracy'].search(self.data)
        if ph_k_degeneracy is None:
            return None

        k_degeneracy_data = {}
        lines = ph_k_degeneracy.group(1).splitlines()
        shr_fact = []
        k_vectors = []
        orig_coords = []
        degenerated = []

        for n in lines:
            if 'WITH SHRINKING FACTORS' in n:
                k = n.split()
                for j in k:
                    if j.isdigit():
                        shr_fact.append(int(j))
            else:
                k = [_f for _f in n.split("   ") if _f]
                if len(k) == 4:
                    orig_coord = k[2].strip().split()
                    orig_coords.append(" ".join(orig_coord))
                    k_coords = [int(i) for i in orig_coord]
                    degenerated.append(int(k[1].replace('.', '')))
                    k_vectors.append(k_coords)

        if shr_fact is None or k_vectors is None:
            raise CRYSTOUT_Error('Invalid format in phonon k-vector degeneracy data!')

        for vi in range(len(k_vectors)):
            norm_coord = []
            for i in range(len(k_vectors[vi])):
                norm_coord.append("%s" % Fraction(k_vectors[vi][i], shr_fact[i]))
            k_degeneracy_data[orig_coords[vi]] = {'bzpoint': " ".join(norm_coord), 'degeneracy': degenerated[vi]}

        return k_degeneracy_data

    def get_elastic(self, pattern):
        constants = []
        const_rows = self.patterns[pattern].findall(self.data)
        if not const_rows:
            return None
        for i_row, row in enumerate(const_rows[0]):
            constants.append([])
            for ec in range(i_row):
                constants[-1].append(constants[ec][i_row])
            # as number width is 9 in EC output, the numbers can glue together, which is undesirable
            # we can not just split a row
            row = row.strip().rjust((6-i_row)*9, ' ')
            constants[-1] += [float(row[i:i + 9]) for i in range(0, 9*(6-i_row), 9)]
        return constants

    def get_effective_elastic_moduli(self):
        moduli = self.patterns['effective_moduli'].findall(self.data)
        if not moduli:
            return [None, None, None, None, None, None, None, None]
        return [float(x) for x in moduli[0].split()]

    def decide_charges(self):
        charges, magmoms = [], []
        atomcharges = self.patterns['charges'].search(self.data)
        atommagmoms = self.patterns['magmoms'].search(self.data)

        if not atomcharges and self.properties_calc:
            atomcharges = self.patterns['charges'].search(self.pdata)
        if not atommagmoms and self.properties_calc:
            atommagmoms = self.patterns['magmoms'].search(self.pdata)

        # obtain formal charges from pseudopotentials
        iatomcharges = self.patterns['icharges'].findall(self.data)
        pseudo_charges = copy.deepcopy(atomic_numbers)
        for i in range(len(iatomcharges)):
            try:
                Z = int(iatomcharges[i][0].strip())
                P = float(iatomcharges[i][1].strip())
            except (ValueError, IndexError):
                raise CRYSTOUT_Error('Error in pseudopotential info!')
            p_element = [key for key, value in pseudo_charges.items() if value == Z]
            if len(p_element):
                pseudo_charges[p_element[0].capitalize()] = P

        symbols = list(pseudo_charges.keys())

        if atomcharges is not None:
            parts = atomcharges.group().split("ATOM    Z CHARGE  SHELL POPULATION")
            chargedata = parts[1].splitlines()
            for i in chargedata:
                if self.patterns['at_str'].match(i):
                    val = i.split()
                    val[1] = val[1].capitalize()
                    val[3] = val[3][:6] # erroneous by stars
                    if val[1] in symbols:
                        val[3] = pseudo_charges[val[1]] - float(val[3])
                    elif val[1] == 'Xx':
                        val[3] = -float(val[3]) # TODO this needs checking
                    else:
                        raise CRYSTOUT_Error('Unexpected atomic symbol: ' + val[1])
                    charges.append(val[3])
            try:
                self.info['structures'][-1].set_initial_charges(charges)
            except ValueError:
                self.warning('Number of atoms and found charges does not match!') # some issues with CRYSTAL03
        else:
            self.warning('No charges available!')

        if atommagmoms is not None:
            parts = atommagmoms.group().split("ATOM    Z CHARGE  SHELL POPULATION")
            chargedata = parts[1].splitlines()
            for i in chargedata:
                if self.patterns['at_str'].match(i):
                    val = i.split()
                    val[3] = val[3][:6] # erroneous by stars
                    magmoms.append(float(val[3]))
            try:
                self.info['structures'][-1].set_initial_magnetic_moments(magmoms)
            except ValueError:
                self.warning('Number of atoms and found magmoms does not match!') # some issues with CRYSTAL03
        else:
            self.warning('No magmoms available!')

    def get_input_and_meta(self, inputdata):
        version = None
        inputdata = re.sub(' PROCESS(.{32})WORKING\n', '', inputdata) # Warning! MPI statuses may spoil valuable data
        v = self.patterns['version'].search(inputdata)
        if v:
            v = v.group().split("\n")
            major, minor = v[0], v[1]
            # beware of MPI inclusions!
            if '*' in major:
                major = major.replace('*', '').strip()
            if '*' in minor:
                minor = minor.replace('*', '').strip()
                if ':' in minor:
                    minor = minor.split(':')[1].split()[0]
                else:
                    minor = minor.split()[1]
            version = major.replace('CRYSTAL', '') + ' ' + minor

        # get input data
        inputdata = inputdata.splitlines()
        keywords = []
        keywords_flag = False
        trsh_line_flag = False
        trsh_line_cnt = 0

        for i in range(len(inputdata)):
            if trsh_line_flag:
                trsh_line_cnt += 1
            if keywords_flag:
                keywords.append(inputdata[i].strip())
            if inputdata[i].strip() in ["CRYSTAL", "SLAB", "POLYMER", "HELIX", "MOLECULE", "EXTERNAL"]:
                keywords_flag = True
                keywords.extend([inputdata[i - 1].strip(), inputdata[i].strip()])
            if inputdata[i].startswith("END"):
                trsh_line_flag = True
                trsh_line_cnt = 0

        if not keywords:
            # self.warning('No d12-formatted input data in the beginning found!')
            return None, None, version

        keywords = keywords[:-trsh_line_cnt]
        comment = keywords[0]
        keywords = "\n".join(keywords)

        return comment, keywords, version

    def decide_finished(self):
        if self.info['duration'] and not 'TTTTTTTTTTTTTTTTTTTTTTTTTTTTTT ERR' in self.data:
            self.info['finished'] = 0x2
        else:
            err = self.data.split(' ERROR **** ')
            if len(err) > 1:
                self.warning('Error: ' + err[1].split('\n')[0] + '!')
            self.info['finished'] = 0x1

    def get_ph_sym_disps(self):
        symdisps = self.patterns['symdisps'].search(self.data)
        if symdisps is None:
            return None, None
        else:
            lines = symdisps.group().splitlines()
            plusminus = False
            if 'NUMERICAL GRADIENT COMPUTED WITH A SINGLE DISPLACEMENT (+-dx) FOR EACH' in self.data:
                plusminus = True

            disps, magnitude = [], 0
            for n in lines:
                r = self.patterns['needed_disp'].search(n)
                if r:
                    disps.append([int(r.group(1)), r.group(2).replace('D', '').lower()])
                    if plusminus:
                        disps.append([int(r.group(1)), r.group(2).replace('D', '-').lower()])
                elif '= ' in n: # TODO CRYSTAL06 !
                    magnitude = float(n.split()[1])
            if magnitude == 0:
                raise CRYSTOUT_Error('Cannot find displacement magnitude in FREQCALC output!')
            if not len(disps):
                raise CRYSTOUT_Error('Cannot find valid displacement data in FREQCALC output!')
            return disps, magnitude

    def get_static_dielectric_tensor(self):
        # TODO
        return "\n VIBRATIONAL CONTRIBUTIONS TO THE STATIC DIELECTRIC TENSOR:\n" in self.data or \
               "\n VIBRATIONAL CONTRIBUTIONS TO THE STATIC POLARIZABILITY TENSOR:\n" in self.data

    def get_bs(self):
        gbasis = {'bs': {}, 'ecp': {}}

        if " ATOM   X(AU)   Y(AU)   Z(AU)  N. TYPE" in self.data:
            bs = self.data.split(" ATOM   X(AU)   Y(AU)   Z(AU)  N. TYPE") # CRYSTAL<14
        else:
            bs = self.data.split(" ATOM  X(AU)  Y(AU)  Z(AU)    NO. TYPE  EXPONENT ") # CRYSTAL14

        if len(bs) == 1:
            if not self.info['input']:
                self.warning('No basis set found!')

            # NB basis set is absent in output, input may be not enough!
            return CRYSTOUT.parse_bs_input(self.info['input'], then=self.correct_bs_ghost)

        # NO BASE FIXINDEX IMPLEMENTED
        bs = bs[-1].split("*******************************************************************************\n", 1)[-1]
        bs = re.sub(' PROCESS(.{32})WORKING\n', '', bs) # Warning! MPI statuses may spoil valuable data
        bs = bs.splitlines()

        atom_order = []
        atom_type = None

        for line in bs:
            if line.startswith(" " * 20): # gau type or exponents
                if line.startswith(" " * 40): # exponents
                    if not atom_type or not len(gbasis['bs'][atom_type]):
                        raise CRYSTOUT_Error('Unexpected or corrupted basis output - gaussian type or atom not given!')
                    line = line.strip()
                    if line[:1] != '-':
                        line = ' ' + line
                    n = 0
                    gaussians = []
                    for s in line:
                        if not n % 10:
                            gaussians.append(' ')
                        gaussians[-1] += s
                        n += 1

                    gaussians = [x for x in map(float, gaussians) if x != 0]
                    # for i in range(len(gaussians)-1, -1, -1):
                    #    if gaussians[i] == 0: gaussians.pop()
                    #    else: break
                    gbasis['bs'][atom_type][-1].append(tuple(gaussians))

                else: # gau type
                    symb = line.split()[-1]

                    if bs_concurrency:
                        atom_type += '1'
                        bs_concurrency = False
                        try:
                            gbasis['bs'][atom_type]
                        except KeyError:
                            gbasis['bs'][atom_type] = []
                        else:
                            raise CRYSTOUT_Error(
                                'More than two different basis sets for one element - not supported case!')
                    gbasis['bs'][atom_type].append([symb])

            else: # atom N or end
                test = line.split()
                if test and test[0] == 'ATOM':
                    continue # C03: can be odd string ATOM  X(AU)  Y(AU)  Z(AU)
                try:
                    float(test[0])
                except (ValueError, IndexError):
                    break # endb, e.g. void space or INFORMATION **** READM2 **** FULL DIRECT SCF (MONO AND BIEL INT) SELECTED

                atom_type = test[1][:2].capitalize()
                if atom_type == 'Xx':
                    atom_type = 'X'
                atom_order.append(atom_type)

                try:
                    gbasis['bs'][atom_type]
                except KeyError:
                    gbasis['bs'][atom_type] = []
                    bs_concurrency = False
                else:
                    bs_concurrency = True

        # PSEUDOPOTENTIALS
        ecp = self.data.split(" *** PSEUDOPOTENTIAL INFORMATION ***")
        if len(ecp) > 1:
            # NO BASE FIXINDEX IMPLEMENTED
            ecp = ecp[-1].split("*******************************************************************************\n", 2)[-2]
            ecp = re.sub(' PROCESS(.{32})WORKING\n', '', ecp) # Warning! MPI statuses may spoil valuable data
            ecp = ecp.splitlines()
            for line in ecp:
                if 'PSEUDOPOTENTIAL' in line:
                    atnum = int(line.split(',')[0].replace('ATOMIC NUMBER', ''))
                    # int(nc.replace('NUCLEAR CHARGE', ''))
                    if 200 < atnum < 1000:
                        atnum = int(str(atnum)[-2:])
                    atom_type = chemical_symbols[atnum]
                    try:
                        gbasis['ecp'][atom_type]
                    except KeyError:
                        gbasis['ecp'][atom_type] = []
                    else:
                        atom_type += '1'
                        try:
                            gbasis['ecp'][atom_type]
                        except KeyError:
                            gbasis['ecp'][atom_type] = []
                        else:
                            raise CRYSTOUT_Error('More than two pseudopotentials for one element - not supported case!')
                else:
                    lines = line.split()
                    try:
                        float(lines[-2])
                    except (ValueError, IndexError):
                        continue
                    else:
                        if 'TMS' in line:
                            gbasis['ecp'][atom_type].append([lines[0]])
                            lines = lines[2:]
                        lines = list(map(float, lines))
                        for i in range(len(lines) // 3):
                            gbasis['ecp'][atom_type][-1].append(
                                tuple([lines[0 + i * 3], lines[1 + i * 3], lines[2 + i * 3]]))

        # sometimes ghost basis set is printed without exponents and we should determine what atom was replaced
        if 'X' in gbasis['bs'] and not len(gbasis['bs']['X']):
            replaced = atom_order[atom_order.index('X') - 1]
            gbasis['bs']['X'] = copy.deepcopy(gbasis['bs'][replaced])

        return self.correct_bs_ghost(gbasis)

    @staticmethod
    def parse_bs_input(text, as_d12=True, then=lambda x: x):
        """
        Note: input must be scanned only if nothing found in output
        input may contain comments (not expected by CRYSTAL, but user anyway can cheat it)
        WARNING the block /* */ comments will fail TODO?
        """
        gbasis = {'bs': {}, 'ecp': {}}

        if not text:
            return gbasis

        comment_signals = '#/*<!'
        bs_sequence = {
            0: 'S',
            1: 'SP',
            2: 'P',
            3: 'D',
            4: 'F',
            5: 'G'
        }
        bs_type = {
            1: 'STO-nG(nd) type ',
            2: '3(6)-21G(nd) type '
        }
        bs_notation = {
            1: 'n-21G outer valence shell',
            2: 'n-21G inner valence shell',
            3: '3-21G core shell',
            6: '6-21G core shell'
        }
        ps_keywords = {
            'INPUT': None,
            'HAYWLC': 'Hay-Wadt large core',
            'HAYWSC': 'Hay-Wadt small core',
            'BARTHE': 'Durand-Barthelat',
            'DURAND': 'Durand-Barthelat'
        }
        ps_sequence = ['W0', 'P0', 'P1', 'P2', 'P3', 'P4']

        read = not as_d12
        read_pseud, read_bs = False, False

        for line in text.splitlines():
            if line.startswith('END'):
                read = True
                continue

            if not read:
                continue

            for s in comment_signals:
                pos = line.find(s)
                if pos != -1:
                    line = line[:pos]
                    break

            parts = line.split()

            if len(parts) == 1 and parts[0].upper() in ps_keywords:
                # pseudo
                try:
                    gbasis['ecp'][atom_type]
                except KeyError:
                    gbasis['ecp'][atom_type] = []
                else:
                    atom_type += '1'
                    try:
                        gbasis['ecp'][atom_type]
                    except KeyError:
                        gbasis['ecp'][atom_type] = []
                    else:
                        raise CRYSTOUT_Error('More than two pseudopotentials for one element - not supported case!')

                if parts[0] != 'INPUT':
                    gbasis['ecp'][atom_type].append(ps_keywords[parts[0].upper()])

            elif len(parts) == 0:
                continue

            else:
                try:
                    [float(x.replace("D", "E")) for x in line.split()] # sanitary check
                except ValueError:
                    read = False
                    continue

            if len(parts) in [2, 3]:
                # what is this ---- atom, exit, ecp exponent or bs exponent?
                if parts[0] == '99' and parts[1] == '0':
                    break # this is ---- exit

                elif '.' in parts[0] or '.' in parts[1]:
                    # this is ---- ecp exponent or bs exponent
                    parts = [float(x.replace("D", "E")) for x in parts]

                    if read_pseud:
                        # distribute exponents into ecp-types according to counter, that we now calculate
                        if distrib in list(ps_indeces_map.keys()):
                            gbasis['ecp'][atom_type].append([ps_indeces_map[distrib]])
                        gbasis['ecp'][atom_type][-1].append(tuple(parts))
                        distrib += 1
                    elif read_bs:
                        # distribute exponents into orbitals according to counter, that we already defined
                        gbasis['bs'][atom_type][-1].append(tuple(parts))

                else:
                    # this is ---- atom
                    if len(parts[0]) > 2:
                        parts[0] = parts[0][-2:]
                    if int(parts[0]) == 0:
                        atom_type = 'X'
                    else:
                        atom_type = chemical_symbols[int(parts[0])]

                    try:
                        gbasis['bs'][atom_type]
                    except KeyError:
                        gbasis['bs'][atom_type] = []
                    else:
                        atom_type += '1'
                        try:
                            gbasis['bs'][atom_type]
                        except KeyError:
                            gbasis['bs'][atom_type] = []
                        else:
                            raise CRYSTOUT_Error(
                                'More than two different basis sets for one element - not supported case!')
                    continue

            elif len(parts) == 5:
                # this is ---- orbital
                gbasis['bs'][atom_type].append([bs_sequence[int(parts[1])]])
                parts = list(map(int, parts[0:3]))

                if parts[0] == 0:
                    # insert from data given in input
                    read_pseud, read_bs = False, True
                # 1 = Pople standard STO-nG (Z=1-54);
                # 2 = Pople standard 3(6)-21G (Z=1-54(18)) + standard polarization functions
                elif parts[0] in bs_type:
                    # pre-defined insert
                    if parts[2] in bs_notation:
                        gbasis['bs'][atom_type][-1].append(bs_type[parts[0]] + bs_notation[parts[2]])
                    else:
                        gbasis['bs'][atom_type][-1].append(bs_type[parts[0]] + 'n=' + str(parts[2]))

            elif 6 <= len(parts) <= 7:
                # this is ---- pseudo - INPUT
                parts.pop(0)
                ps_indeces = list(map(int, parts))
                ps_indeces_map = {}
                accum = 1
                for c, n in enumerate(ps_indeces):
                    if n == 0:
                        continue
                    ps_indeces_map[accum] = ps_sequence[c]
                    accum += n
                distrib = 1
                read_pseud, read_bs = True, False

        return then(gbasis)

    def correct_bs_ghost(self, gbasis):
        # ghost cannot be in pseudopotential
        atoms = []
        for atom in self.info['structures'][-1].get_chemical_symbols():
            if not atom in atoms:
                atoms.append(atom)

        for k, v in gbasis['bs'].items():
            # sometimes no BS for host atom is printed when it is replaced by Xx: account it
            if not len(v) and k != 'X' and 'X' in gbasis['bs']:
                gbasis['bs'][k] = copy.deepcopy(gbasis['bs']['X'])

        return gbasis

    def decide_method(self):

        # Hamiltonian part
        hamiltonian_parts = { # TODO
            'DIRAC-SLATER LDA': {'name': 'LDA', 'type': 0x1},
            'PERDEW-ZUNGER': {'name': 'PZ_LDA', 'type': 0x1},
            'VOSKO-WILK-NUSAIR': {'name': 'WVN_LDA', 'type': 0x1},
            'PERDEW-WANG LSD': {'name': 'PW_LDA', 'type': 0x1},
            'VON BARTH-HEDIN': {'name': 'VBH_LDA', 'type': 0x1},

            'PERDEW-WANG GGA': {'name': 'PW_GGA', 'type': 0x2},
            'BECKE': {'name': 'B_GGA', 'type': 0x2},
            'LEE-YANG-PARR': {'name': 'LYP_GGA', 'type': 0x2},
            'PERDEW-BURKE-ERNZERHOF': {'name': 'PBE_GGA', 'type': 0x2},
            'SOGGA': {'name': 'SOGGA', 'type': 0x2},
            'PERDEW86': {'name': 'P86_GGA', 'type': 0x2},
            'PBEsol': {'name': 'PBESOL_GGA', 'type': 0x2},
            'WILSON-LEVY': {'name': 'WL_GGA', 'type': 0x2},
            'WU-COHEN GGA': {'name': 'WC_GGA', 'type': 0x2},
        }
        exch, corr = '', ''

        if ' HARTREE-FOCK HAMILTONIAN\n' in self.data:
            self.info['H'] = 'Hartree-Fock'
            self.info['H_types'].append(0x5)

        elif ' (EXCHANGE)[CORRELATION] FUNCTIONAL:' in self.data:
            exch, corr = self.data.split(' (EXCHANGE)[CORRELATION] FUNCTIONAL:', 1)[-1].split("\n", 1)[0].split(')[')
            exch = exch.replace("(", "")
            corr = corr.replace("]", "")

            try:
                self.info['H_types'].append(hamiltonian_parts[exch]['type'])
                exch = hamiltonian_parts[exch]['name']
            except KeyError:
                self.warning('Unknown potential: %s' % exch)
            try:
                if not hamiltonian_parts[corr]['type'] in self.info['H_types']:
                    self.info['H_types'].append(hamiltonian_parts[corr]['type'])
                corr = hamiltonian_parts[corr]['name']
            except KeyError:
                self.warning('Unknown potential: %s' % corr)

            if exch == 'PBE_GGA' and corr == 'PBE_GGA':
                self.info['H'] = 'PBE'
            elif exch == 'PBESOL_GGA' and corr == 'PBESOL_GGA':
                self.info['H'] = 'PBEsol'
            else:
                self.info['H'] = "%s/%s" % (exch, corr)

        elif '\n THE CORRELATION FUNCTIONAL ' in self.data:
            corr = self.data.split('\n THE CORRELATION FUNCTIONAL ', 1)[-1].split("\n", 1)[0].replace("IS ACTIVE",
                                                                                                      "").strip()
            name = corr
            try:
                name = hamiltonian_parts[corr]['name']
                self.info['H_types'].append(hamiltonian_parts[corr]['type'])
            except KeyError:
                self.warning('Unknown potential: %s' % corr)
            self.info['H'] = "%s (pure corr.)" % name

        elif '\n THE EXCHANGE FUNCTIONAL ' in self.data:
            exch = self.data.split('\n THE EXCHANGE FUNCTIONAL ', 1)[-1].split("\n", 1)[0].replace("IS ACTIVE",
                                                                                                   "").strip()
            name = exch
            try:
                name = hamiltonian_parts[exch]['name']
                self.info['H_types'].append(hamiltonian_parts[exch]['type'])
            except KeyError:
                self.warning('Unknown potential: %s' % exch)
            self.info['H'] = "%s (pure exch.)" % name

        if '\n HYBRID EXCHANGE ' in self.data:
            self.info['H_types'].append(0x4)
            hyb = self.data.split('\n HYBRID EXCHANGE ', 1)[-1].split("\n", 1)[0].split()[-1]
            hyb = int(math.ceil(float(hyb)))

            if hyb == 25 and self.info['H'] == 'PBE':
                self.info['H'] = 'PBE0'
            elif hyb == 20 and exch == 'B_GGA' and corr == 'LYP_GGA':
                self.info['H'] = 'B3LYP'
            elif hyb == 20 and exch == 'B_GGA' and corr == 'PW_GGA':
                self.info['H'] = 'B3PW'
            else:
                ham = self.info['H'].split('/')
                ham[0] += " (+" + str(hyb) + "%HF)"
                self.info['H'] = '/'.join(ham)

        if not self.info['H']:
            self.warning('Potential not found!')
            self.info['H'] = "unknown"

        # Spin part
        if ' TYPE OF CALCULATION :  UNRESTRICTED OPEN SHELL' in self.data:
            self.info['spin'] = True
            if '\n ALPHA-BETA ELECTRONS LOCKED TO ' in self.data:
                spin_info = self.data.split('\n ALPHA-BETA ELECTRONS LOCKED TO ', 1)[-1].split("\n", 1)[0].replace(
                    'FOR', '').split()
                cyc = int(spin_info[1])
                if self.info['ncycles'] and self.info['ncycles'][0] < cyc:
                    self.info['lockstate'] = int(spin_info[0])

        # K-points part
        if '\n SHRINK. FACT.(MONKH.) ' in self.data:
            kset = self.data.split('\n SHRINK. FACT.(MONKH.) ', 1)[-1].split()
            if len(kset) < 4:
                self.warning('Unknown k-points format!')
            self.info['k'] = "x".join(kset[:3])

        # Perturbation part
        if "* *        COUPLED-PERTURBED KOHN-SHAM CALCULATION (CPKS)         * *" in self.data:
            self.info['techs'].append('perturbation: analytical')
        elif "\n F F F F F F F F F F F F F F F F F F F F F F F F F F F F F F F F F F F F F F F\n" in self.data:
            self.info['techs'].append('perturbation: numerical')

        # Tolerances part
        if 'COULOMB OVERLAP TOL         (T1)' in self.data:
            t = [int(self.data.split('COULOMB OVERLAP TOL         (T1)', 1)[-1].split("\n", 1)[0].split('**')[-1]),
                 int(self.data.split('COULOMB PENETRATION TOL     (T2)', 1)[-1].split("\n", 1)[0].split('**')[-1]),
                 int(self.data.split('EXCHANGE OVERLAP TOL        (T3)', 1)[-1].split("\n", 1)[0].split('**')[-1]),
                 int(self.data.split('EXCHANGE PSEUDO OVP (F(G))  (T4)', 1)[-1].split("\n", 1)[0].split('**')[-1]),
                 int(self.data.split('EXCHANGE PSEUDO OVP (P(G))  (T5)', 1)[-1].split("\n", 1)[0].split('**')[-1])]
            for n, i in enumerate(t):
                if i <= 0:
                    continue
                self.warning('Tolerance T%s > 0, assuming default.' % n)
                if n == 4:
                    t[n] = -12
                else:
                    t[n] = -6

            self.info['tol'] = t
            self.info['techs'].append("biel.intgs 10<sup>" + ",".join(map(str, t)) + "</sup>") # TODO

        # Speed-up techniques part
        if '\n WEIGHT OF F(I) IN F(I+1)' in self.data:
            f = int(self.data.split('\n WEIGHT OF F(I) IN F(I+1)', 1)[-1].split('%', 1)[0])
            # TODO CRYSTAL14 default fmixing!
            if 0 < f <= 25:
                self.info['techs'].append('mixing<25%')
            elif 25 < f <= 50:
                self.info['techs'].append('mixing 25-50%')
            elif 50 < f <= 75:
                self.info['techs'].append('mixing 50-75%')
            elif 75 < f <= 90:
                self.info['techs'].append('mixing 75-90%')
            elif 90 < f:
                self.info['techs'].append('mixing>90%')

        if ' ANDERSON MIX: BETA= ' in self.data:
            self.info['techs'].append('mixing by anderson')

        if '\n % OF FOCK/KS MATRICES MIXING WHEN BROYDEN METHOD IS ON' in self.data:
            # mixing percentage, parameter and number of activation cycle
            f = int(
                self.data.split('\n % OF FOCK/KS MATRICES MIXING WHEN BROYDEN METHOD IS ON', 1)[-1].split("\n", 1)[0])
            f2 = float(self.data.split('\n WO PARAMETER(D.D. Johnson, PRB38, 12807,(1988)', 1)[-1].split("\n", 1)[0])
            f3 = int(
                self.data.split('\n NUMBER OF SCF ITERATIONS AFTER WHICH BROYDEN METHOD IS ACTIVE', 1)[-1].split("\n",
                                                                                                                 1)[0])
            value = ""
            if 0 < f <= 25:
                value = 'broyden<25%'
            elif 25 < f <= 50:
                value = 'broyden 25-50%'
            elif 50 < f <= 75:
                value = 'broyden 50-75%'
            elif 75 < f <= 90:
                value = 'broyden 75-90%'
            elif 90 < f:
                value = 'broyden>90%'

            if round(f2, 4) == 0.0001:
                value += ' (std.)' # broyden parameter
            else:
                value += ' (' + str(round(f2, 5)) + ')'
            if f3 < 5:
                value += ' start'
            else:
                value += ' defer.'

            self.info['techs'].append(value)

        if '\n EIGENVALUE LEVEL SHIFTING OF ' in self.data:
            f = float(
                self.data.split('\n EIGENVALUE LEVEL SHIFTING OF ', 1)[-1].split("\n", 1)[0].replace('HARTREE', ''))
            if 0 < f <= 0.5:
                self.info['techs'].append('shifter<0.5au')
            elif 0.5 < f <= 1:
                self.info['techs'].append('shifter 0.5-1au')
            elif 1 < f <= 2.5:
                self.info['techs'].append('shifter 1-2.5au')
            elif 2.5 < f:
                self.info['techs'].append('shifter>2.5au')

        if '\n FERMI SMEARING - TEMPERATURE SMEARING OF FERMI SURFACE ' in self.data:
            f = float(
                self.data.split('\n FERMI SMEARING - TEMPERATURE SMEARING OF FERMI SURFACE ', 1)[-1].split("\n", 1)[0])
            self.info['smear'] = f

            if 0 < f <= 0.005:
                self.info['techs'].append('smearing<0.005au')
            elif 0.005 < f <= 0.01:
                self.info['techs'].append('smearing 0.005-0.01au')
            elif 0.01 < f:
                self.info['techs'].append('smearing>0.01au')

    def get_duration(self):
        starting = self.patterns['starting'].search(self.data)
        ending = self.patterns['ending'].search(self.data)

        if ending is None and self.properties_calc:
            ending = self.patterns['ending'].search(self.pdata)

        if starting is not None and ending is not None:
            starting = starting.group(1).replace("DATE", "").replace("TIME", "").strip()[:-2]
            ending = ending.group(1).replace("DATE", "").replace("TIME", "").strip()[:-2]

            start = time.strptime(starting, "%d %m %Y  %H:%M:%S")
            end = time.strptime(ending, "%d %m %Y  %H:%M:%S")
            duration = "%2.2f" % ((time.mktime(end) - time.mktime(start)) / 3600)
        else:
            self.warning("No timings available!")
            duration = None

        return duration

    def decide_scfdata(self):
        if self.info['input'] is not None and "ONELOG" in self.info['input']:
            self.warning("ONELOG keyword is not supported!")
            return

        convergdata = []
        ncycles = []
        energies = []
        criteria = [[], [], [], []]
        optgeom = []
        zpcycs = self.patterns['cyc'].findall(self.data)

        if zpcycs is not None:
            for i in zpcycs:
                numdata = i.split(" DETOT ")
                num = numdata[1].split()
                try:
                    f = float(num[0]) * Hartree
                except ValueError:
                    f = 0
                if f != 0 and not math.isnan(f):
                    convergdata.append(int(math.floor(math.log(abs(f), 10))))

        else:
            self.warning('SCF not found!')

        self.info['convergence'] = convergdata

        enes = self.patterns['enes'].findall(self.data)

        if enes is not None:
            for i in enes:
                i = i.replace("DFT)(AU)(", "").replace("HF)(AU)(", "").split(")")
                s = i[0]
                if "*" in s:
                    s = 1000
                else:
                    s = int(s)
                ncycles.append(s)
                ene = i[1].split("DE")[0].strip()

                try:
                    ene = float(ene) * Hartree
                except ValueError:
                    ene = None
                energies.append(ene)

        n = 0
        for cr in [self.patterns['t1'], self.patterns['t2'], self.patterns['t3'], self.patterns['t4']]:
            kd = cr.findall(self.data)
            if kd is not None:
                for i in kd:
                    p = i.split("THRESHOLD")
                    p2 = p[1].split("CONVERGED")
                    try:
                        k = float(p[0]) - float(p2[0])
                    except ValueError:
                        k = 999
                    if k < 0:
                        k = 0
                    criteria[n].append(k)
                n += 1

        # print len(criteria[0]), len(criteria[1]), len(criteria[2]), len(criteria[3]), len(energies)
        # ORDER of values: geometry, then energy, then tolerances
        if criteria[-1]:
            if len(criteria[0]) - len(criteria[2]) == 1 and len(criteria[1]) - len(
                    criteria[3]) == 1: # if no restart, then 1st cycle has no treshold t3 and t4
                criteria[2].insert(0, 0)
                criteria[3].insert(0, 0)

            if len(criteria[0]) - len(criteria[2]) == 2 and len(criteria[1]) - len(
                    criteria[3]) == 2: # convergence achieved without t3 and t4 at the last cycle
                criteria[2].insert(0, 0)
                criteria[2].append(criteria[2][-1])
                criteria[3].insert(0, 0)
                criteria[3].append(criteria[3][-1])

            if len(criteria[0]) - len(energies) == 1:
                self.warning(
                    'Energy was not printed at the intermediate step, so the correspondence is partially lost!')
                energies.insert(0, energies[0])
                ncycles.insert(0, ncycles[0])

            if len(criteria[1]) - len(criteria[2]) > 1:
                self.warning('Number of the optgeom tresholds is inconsistent!')

            if len(criteria[2]) > len(energies):
                self.warning(
                    'Energy was not printed at the intermediate step, so the correspondence is partially lost!')

            lengths = [len(criteria[0]), len(criteria[1]), len(criteria[2]), len(criteria[3]), len(energies)]
            for i in range(max(lengths)):
                optgeom.append([
                    criteria[0][i] if i < lengths[0] else None,
                    criteria[1][i] if i < lengths[1] else None,
                    criteria[2][i] if i < lengths[2] else None,
                    criteria[3][i] if i < lengths[3] else None,
                    energies[i]    if i < lengths[4] else None
                ])

        self.info['ncycles'] = ncycles
        self.info['optgeom'] = optgeom

    def get_zpe(self):
        if "\n E0            :" in self.data:
            zpe = self.data.split("\n E0            :")[1].split("\n", 1)[0].split()[0] # AU
            try:
                zpe = float(zpe)
            except ValueError:
                return None
            else:
                return zpe * Hartree
        else:
            return None

    def get_td(self):
        td = {'t': [], 'p': [], 'pv': [], 'ts': [], 'et': [], 'C': [], 'S': []}
        t = self.patterns['T'].findall(self.data)

        if t is not None:
            for i in t:
                td['t'].append(float(i[0]))
                td['p'].append(float(i[1]))

        pv = self.patterns['pv'].findall(self.data)

        if pv is not None:
            for i in pv:
                td['pv'].append(float(i.split()[1])) # EV/CELL

        ts = self.patterns['ts'].findall(self.data)

        if ts is not None:
            for i in ts:
                i = i.split()[1]
                try:
                    i = float(i)
                    if math.isnan(i):
                        i = 0.0
                except ValueError:
                    i = 0.0
                td['ts'].append(float(i)) # EV/CELL

        et = self.patterns['et'].findall(self.data)

        if et is not None:
            for i in et:
                i = i.split()[1]
                try:
                    i = float(i)
                    if math.isnan(i):
                        i = 0.0
                except ValueError:
                    i = 0.0
                td['et'].append(float(i)) # EV/CELL

        s = self.patterns['entropy'].findall(self.data)

        if s is not None:
            for i in s:
                i = i.split()[2]
                try:
                    i = float(i)
                    if math.isnan(i):
                        i = 0.0
                except ValueError:
                    i = 0.0
                td['S'].append(float(i)) # J/(MOL*K)

        c = self.patterns['C'].findall(self.data)

        if c is not None:
            for i in c:
                i = i.split()[2]
                try:
                    i = float(i)
                    if math.isnan(i):
                        i = 0.0
                except ValueError:
                    i = 0.0
                td['C'].append(float(i)) # J/(MOL*K)

        if td['t'] and td['pv'] and td['ts'] and td['et']:
            return td

        else:
            self.warning('Errors in thermodynamics!')
            return None
