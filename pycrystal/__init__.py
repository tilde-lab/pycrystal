
import sys

from .output import CRYSTOUT, CRYSTOUT_Error
from .gaussian_basis import parse_bs
from .basis_library import download_basis_library


MIN_PY_VER = (3, 5)

assert sys.version_info >= MIN_PY_VER, "Python version must be >= {}".format(MIN_PY_VER)