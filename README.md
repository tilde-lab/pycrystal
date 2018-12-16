CRYSTAL ab initio code utilities
==========

```
import os, sys
from pprint import pprint

from pycrystal import CRYSTOUT

if __name__ == "__main__":
    try:
        sys.argv[1] and os.path.exists(sys.argv[1])
    except (IndexError, OSError):
        sys.exit("USAGE: <script> <file>")

    assert CRYSTOUT.acceptable(sys.argv[1])
    result = CRYSTOUT(sys.argv[1])
    pprint(result.info)
```

Also, for any basis set taken from [EMSL](https://bse.pnl.gov) in Gaussian'94 format:

```
import os, sys

from pycrystal import parse_bs

if __name__ == "__main__":
    try:
        sys.argv[1] and os.path.exists(sys.argv[1])
    except (IndexError, OSError):
        sys.exit("USAGE: <script> <file>")

    content = open(sys.argv[1]).read()
    for atom in parse_bs(content):
        print atom.crystal_input()
```
