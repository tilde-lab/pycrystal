
import re
from pprint import pprint
import logging

import requests
from bs4 import BeautifulSoup


chemical_symbols = ['X',
'H',                                                                                                                                                                                      'He',
'Li', 'Be',                                                                                                                                                 'B',  'C',  'N',  'O',  'F',  'Ne',
'Na', 'Mg',                                                                                                                                                 'Al', 'Si', 'P',  'S',  'Cl', 'Ar',
'K',  'Ca',                                                                                     'Sc', 'Ti', 'V',  'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
'Rb', 'Sr',                                                                                     'Y',  'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I',  'Xe',
'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W',  'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn',
'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U',  'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og']


def mine_unito():

    _els2paperids = {}
    _paperids2bib = {}

    data_url_elements = 'http://www.crystal.unito.it/elementi/node1.html'
    data_url_articles = 'http://www.crystal.unito.it/elementi/node2.html'
    regex = re.compile(r"\<.*?\>", re.IGNORECASE)

    r = requests.get(data_url_elements)
    assert r.status_code == 200

    soup = BeautifulSoup(r.content, 'html.parser')
    for item in soup.find_all('li'):

        el_data = [
            item.replace('\r\n', '') for item in str(item.contents).split('<br/>')
            if '<a name="z' in item or '<a href="node2' in item
        ]

        last_key = None

        for item in el_data:
            if '<a name="z' in item:
                try:
                    key = item.split('(')[1].split(')')[0]
                except IndexError:
                    if 'RUTHENIUM' in item:
                        key = 44
                else:
                    key = int(key.lower().split('z=')[-1])

                last_key = chemical_symbols[key]
                _els2paperids[last_key] = []

            elif '<a href="node2' in item:
                for anchor in item.split('<a href="node2.html#')[1:]:
                    anchor = anchor.split('">')[0]
                    assert last_key
                    _els2paperids[last_key].append(anchor)


    r = requests.get(data_url_articles)
    assert r.status_code == 200

    articles_html = r.content.decode('utf-8').split('<DL COMPACT>')[1].split('<ADDRESS>')[0]

    for item in articles_html.split('<P></P><DT>'):
        item = item.replace('&nbsp;', ' ').replace('\r', '').replace('\n', '')
        anchor =      item.split('<DD>')[0]
        pub_content = item.split('<DD>')[1]

        if '``' not in pub_content or "''," not in pub_content:
            logging.warning("CANNOT PARSE >>> %s" % pub_content)

            # unparsable cases
            if 'R. Dovesi, C. Roetti, M. Caus&#224; and C. Pisani' in pub_content and '<EM>Structure and Reactivity of Surfaces</EM>, edited by C. Morterra' in pub_content:
                _paperids2bib['to100'] = ('R. Dovesi, C. Roetti, M. Causa and C. Pisani', 'Ab initio study of the periodic carbon monoxide adsorption on the basal plane of alpha-alumina', 1989, 'Structure and Reactivity of Surfaces, Elsevier Science, Amsterdam')
            elif 'R. Orlando, C. Pisani, C. Roetti and E. Stefanovich' in pub_content and 'Nordkirchen, Germany' in pub_content:
                _paperids2bib['to168'] = ('R. Orlando, C. Pisani, C. Roetti and E. Stefanovich', 'Ab initio Hartree-Fock study of tetragonal and cubic phases of zirconium dioxide', 1992, 'Defects in insulating materials, 630-632, World Scientific, Nordkirchen, Germany')
            elif 'Ab-initio approaches to the quantum-mechanical treatment of' in pub_content:
                _paperids2bib['to230'] = ('C. Pisani', 'Ab-initio approaches to the quantum-mechanical treatment of periodic systems', 1996, 'Quantum-Mechanical Ab-initio Calculation of the Properties of Crystalline Materials, Springer Verlag, Berlin')
            elif 'CRYSTAL88, An ab initio all-electron LCAO-Hartree-Fock program' in pub_content:
                _paperids2bib['to98'] = ('R. Dovesi, C. Pisani, C. Roetti, M. Causa and V.R. Saunders', 'CRYSTAL88, An ab initio all-electron LCAO-Hartree-Fock program for periodic systems', 1989, 'QCPE Pgm N. 577, Quantum Chemistry Program Exchange, Indiana University, Bloomington, Indiana')
            continue

        anchor = anchor.split('NAME="')[-1].split('">')[0]

        authors = pub_content.split('``')[0].strip().replace('&#246;', 'oe').replace('&#224;', 'a').replace('&#225;', 'a').replace('&#228;', 'a').replace('&#230;', 'ae').replace('&#252;', 'u').replace('&#233;', 'e')
        authors = authors.strip(',')

        title = pub_content.split('``', 1)[1].split("'',")[0].replace('  ', ' ')
        title = regex.sub('', title)

        pubdata = pub_content.split('``', 1)[1].split("'',", 1)[1].replace('  ', ' ').strip()
        pubyear = pubdata.split('(')[-1].split(')')[0]
        try: pubyear = int(pubyear)
        except ValueError:
            logging.warning("CANNOT GET YEAR FROM: %s" % pubdata)
        pubdata = regex.sub('', pubdata.split('(')[0])

        _paperids2bib[anchor] = (authors, title, pubyear, pubdata)

    for el in _els2paperids:
        for paperid in _els2paperids[el]:
            assert paperid in _paperids2bib

    return _els2paperids, _paperids2bib


if __name__ == "__main__":

    _els2paperids, _paperids2bib = mine_unito()

    pprint(_els2paperids)
    pprint(_paperids2bib)
