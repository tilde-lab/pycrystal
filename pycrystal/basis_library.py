
import re
import logging

import requests
from bs4 import BeautifulSoup

from pycrystal import CRYSTOUT
from ase.data import atomic_names, chemical_symbols


altnames = {'sulfur': 'sulphur', 'molybdenum': 'molibdenum', 'technetium': 'technecium'} # NB spelling
known_page_names = [name.lower() for name in atomic_names[1:]]
known_page_names = [altnames[name] if name in altnames else name for name in known_page_names]
ref_page_names = [None] + known_page_names[:] # PS from zero-th element
linebreak = re.compile("(\r\n|\n){2}")


def download_basis_library():

    library = {}

    while True:
        if not known_page_names:
            break
        page = known_page_names.pop(0)

        r = requests.get('http://www.crystal.unito.it/Basis_Sets/%s.html' % page)
        if r.status_code != 200:
            logging.warning("NO PAGE FOR %s" % page)
            continue

        soup = BeautifulSoup(r.content, 'html.parser')

        for a in soup.find_all('a'):
            anchor = a.get('href', '')
            if anchor.startswith('http') and anchor.endswith('.html') and '#' not in anchor:
                logging.warning("PAGE %s HAS AN EXTERNAL LINK: %s" % (page, anchor))
                known_page_names.append(
                    anchor.split('/')[-1][:-5]
                )

        for basis in soup.find_all('pre'):
            title = basis.findPrevious('p').text.strip()
            if not title:
                title = basis.findPrevious('font').text.strip()

            parts = [item for item in linebreak.split(basis.text) if len(item) > 2]
            page = page.split('_')[0] # PS such as http://www.crystal.unito.it/Basis_Sets/oxygen_baranek.html

            # Correct mis-formats in the BS library at the CRYSTAL website
            if page == 'sulphur' and '10.1002/jcc.23153' in parts[1]:
                # "1 0" -> "1.0"
                parts[0] = parts[0].replace("\r\n0 3 1 0.0 1 0\r\n 0.5207010100 1.00000000000000", "\r\n0 3 1 0.0 1.0\r\n 0.5207010100 1.00000000000000")

            elif page == 'sulphur' and '10.1002/jcc.26013' in parts[1]:
                # "1 0" -> "1.0"
                parts[0] = parts[0].replace("\r\n0 3 1 0.0 1 0\r\n  0.4107010100     1.0000000000000", "\r\n0 3 1 0.0 1.0\r\n  0.4107010100     1.0000000000000")

            elif page == 'titanium' and 'Mahmoud' in parts[1]:
                # remove "Ti"
                parts[0] = parts[0].replace("Ti\r\n22 9\r\n", "22 9\r\n")

            elif page == 'bismuth' and 'weihrich' in title:
                # remove comment
                parts[0] = parts[0].replace("ECP modified from Hay and Wadt, JCP 82, 284 (1985)", "")

            elif page == 'mercury' and 'weihrich' in title:
                # remove comments
                parts[0] = parts[0].replace("ECP modified from Hay and Wadt, JCP 82, 1985", "")

            elif page == 'thallium' and 'Bachhuber' in title:
                # fix INPUT
                parts[0] = parts[0].replace("13. 6 5 6 6 6 0 0", "13. 6 5 6 6 6 0")

            elif page == 'oxygen' and 'corno' in title:
                # remove comments
                parts[0] = parts[0].replace("same as gatti_1994", "").replace("gatti_1994 modified", "")

            elif page == 'plutonium' and '_NO_G_' in title:
                # fix Pu ECP format
                parts[0] = parts[0].replace("294 11", "294 9")

            elif page == 'polonium' and 'TZVP_rev2' in title:
                # fix Po
                parts[0] = parts[0].replace("282 12", "284 12")

            # NB. sometimes the comments get included afterwards

            expected_element = chemical_symbols[ref_page_names.index(page)]

            parsed = CRYSTOUT.parse_bs_input(parts[0], as_d12=False)
            gbasis = dict(
                data=parts[0].strip(),
                meta=" ".join(parts[1:]).replace("\n", " ").replace("\r", "").strip(),
                title=title
            )
            element = list(parsed['bs'].keys())[0]
            assert expected_element == element, "%s is on the page of %s" % (element, expected_element)

            library.setdefault(element, []).append(gbasis)

    return library


if __name__ == "__main__":

    import os
    import pickle
    from ase.data import chemical_symbols

    expect_absent = {'He', 'Kr', 'Xe', 'At', 'Rn', 'Fr', 'Ra', 'Fm', 'Md', 'No', 'Lr', 'Rf'}
    CACHE_FILE = 'bs_library_test.pkl'

    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'rb') as f:
            bs_library = pickle.load(f)
    else:
        bs_library = download_basis_library()
        with open(CACHE_FILE, 'wb') as f:
            pickle.dump(bs_library, f, protocol=2)

    for el in chemical_symbols[1:105]: # until Rf incl.
        if el not in bs_library and el not in expect_absent:
            raise RuntimeError
