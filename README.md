CRYSTAL ab initio code utilities
==========

![PyPI](https://img.shields.io/pypi/v/pycrystal.svg?style=flat)

![CRYSTAL ab initio code with the LCAO Gaussian basis sets, by Turin university](https://raw.githubusercontent.com/tilde-lab/pycrystal/master/crystal-dft-redrawn-logo.svg "CRYSTAL17 ab initio LCAO code with the Gaussian basis sets, Torino")

Intro
------

The [CRYSTAL](http://www.crystal.unito.it) is an _ab initio_ solid state modeling suite employing the Gaussian basis sets in the LCAO framework. The `pycrystal` Python utilities are good for:

* quick logs parsing, getting the maximum information, and presenting it in a systematic machine-readable way
* preparing and handling the Gaussian LCAO basis sets, based on the EMSL and own CRYSTAL libraries

All the popular versions of the CRYSTAL code are supported (CRYSTAL03, CRYSTAL06, CRYSTAL09, CRYSTAL14, and CRYSTAL17).

The `pycrystal` was tested on about 20k in-house simulation logs for about 700 distinct materials systems, produced with the different CRYSTAL versions. Its development was initiated in 2009 by [Maxim Losev](https://github.com/mlosev) at the quantum chemistry chair, chemistry dept. of St. Petersburg State University (Russia) under supervision of Professor Robert Evarestov.

Installation
------

`pip install pycrystal`

Usage
------

Parsing is done as follows:

```python
import os, sys
from pprint import pprint

from pycrystal import CRYSTOUT

try:
    sys.argv[1] and os.path.exists(sys.argv[1])
except (IndexError, OSError):
    sys.exit("USAGE: <script> <file>")

assert CRYSTOUT.acceptable(sys.argv[1])
result = CRYSTOUT(sys.argv[1])
pprint(result.info)
```

Also, for any basis set taken from [EMSL](https://bse.pnl.gov) in Gaussian'94 format:

```python
import os, sys

from pycrystal import parse_bs

try:
    sys.argv[1] and os.path.exists(sys.argv[1])
except (IndexError, OSError):
    sys.exit("USAGE: <script> <file>")

content = open(sys.argv[1]).read()
for atom in parse_bs(content):
    print(atom.crystal_input())
```

To deal with the CRYSTAL (University of Turin) online BS library:

```python
from pycrystal import download_basis_library

library = download_basis_library()
print(library['Bi'])
```

Related work
------

There is another Python parser [ejplugins](https://github.com/chrisjsewell/ejplugins) for CRYSTAL14 and CRYSTAL17 by [Chris Sewell](https://github.com/chrisjsewell) (Imperial College London, UK). The comparison was done using `cmp_unito_crystal_parsers.py` script on the above-mentioned 20k logs, the results are as follows:

* the final total energies and atomic structures are the same in more than 99% cases
* `pycrystal` supports slightly more CRYSTAL features than `ejplugins`
* `pycrystal` is more lightweight than `ejplugins` and has less dependencies
* performance is nearly the same
