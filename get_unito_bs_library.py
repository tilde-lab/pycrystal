#!/usr/bin/env python
"""
This script downloads the entire
CRYSTAL basis sets library
into an SQLite database file
"""
import os
import json
import logging
import re
import sqlite3

import requests
from bs4 import BeautifulSoup

from pycrystal import CRYSTOUT


DEFAULT_SQLITE_DB = os.path.dirname(os.path.abspath(__file__)) + os.sep + 'bs_library.db'
assert not os.path.exists(DEFAULT_SQLITE_DB)

conn = sqlite3.connect(DEFAULT_SQLITE_DB)
handle = conn.cursor()
handle.execute("""CREATE TABLE lcao (id INTEGER PRIMARY KEY AUTOINCREMENT, key TEXT, value BLOB);""")

# NB irregular spelling: sulfur -> sulphur, molybdenum -> molibdenum, technetium -> technecium
known_page_names = ["hydrogen", "helium", "lithium", "beryllium", "boron", "carbon", "nitrogen", "oxygen", "fluorine", "neon", "sodium", "magnesium", "aluminium", "silicon", "phosphorus", "sulphur", "chlorine", "argon", "potassium", "calcium", "scandium", "titanium", "vanadium", "chromium", "manganese", "iron", "cobalt", "nickel", "copper", "zinc", "gallium", "germanium", "arsenic", "selenium", "bromine", "krypton", "rubidium", "strontium", "yttrium", "zirconium", "niobium", "molibdenum", "technetium", "ruthenium", "rhodium", "palladium", "silver", "cadmium", "indium", "antimony", "tellurium", "iodine", "xenon", "caesium", "barium", "lanthanum", "cerium", "praseodymium", "neodymium", "promethium", "samarium", "europium", "gadolinium", "terbium", "dysprosium", "holmium", "erbium", "thulium", "ytterbium", "lutetium", "hafnium", "tantalum", "tungsten", "rhenium", "osmium", "iridium", "platinum", "gold", "mercury", "thallium", "lead", "bismuth", "polonium", "astatine", "radon", "francium", "radium", "actinium", "thorium", "protactinium", "uranium", "neptunium", "plutonium", "americium", "curium", "berkelium", "californium", "einsteinium", "fermium", "mendelevium", "nobelium", "lawrencium", "roentgenium"]

linebreak = re.compile("(\r\n|\n){2}")


def get_new_page():
    if not known_page_names:
        return None
    return known_page_names.pop(0)


while True:
    page = get_new_page()
    if not page:
        break

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

        # FIXME! Two mis-formats in the BS library at the CRYSTAL website (reported)
        if page == 'sulphur' and '10.1002/jcc.23153' in parts[1]:
            continue
        elif page == 'titanium' and 'Mahmoud' in parts[1]:
            continue

        parsed = CRYSTOUT.parse_bs_input(parts[0], as_d12=False)
        gbasis = {}
        gbasis['data'] = parts[0]
        gbasis['meta'] = str(" ".join(parts[1:]).replace("\n", " ").replace("\r", "").strip().encode('ascii', 'ignore'))
        gbasis['title'] = title

        element = list(parsed['bs'].keys())[0]
        handle.execute("INSERT INTO lcao(key, value) VALUES (?, ?);", (element, json.dumps(gbasis)))

conn.commit()
conn.close()
