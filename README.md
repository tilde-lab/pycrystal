CRYSTAL ab initio code utilities
==========

```
import os
import sys
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
