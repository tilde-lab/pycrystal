
import os
import sys
import time
import logging
import json
import random

from unito_parser import mine_unito
from crossref_parser import mine_doi
from scihub_parser import retrieve_paper


logging.basicConfig(level=logging.INFO)


if __name__ == "__main__":

    if not os.path.exists('data'):
        os.makedirs('data')

    data_step_two = 'data/els2data.json'
    assert not os.path.exists(data_step_two)

    try:
        data_step_one = sys.argv[1]
    except IndexError:
        data_step_one = 'data/els2bib.json'
        output = {}
        output['els2paperids'], output['paperids2bib'] = mine_unito()
        f = open(data_step_one, 'w')
        f.write(json.dumps(output, indent=4))
        f.close()

    if os.path.exists(data_step_one):
        f = open(data_step_one)
        els2bib = json.loads(f.read())
        f.close()
    else:
        raise RuntimeError

    paperids2data = {}

    for key, item in els2bib['paperids2bib'].items():

        result = mine_doi(*item)
        if not result:
            logging.critical('DOI ERROR WITH %s (%s)' % (key, item))
            continue

        doi, authors, title, true_pubdata, pubyear = result
        file_name = 'data/' + key + '.pdf'

        if os.path.exists(file_name):
            paperids2data[key] = (file_name, doi, authors, title, true_pubdata, pubyear)
            continue

        retrieved = retrieve_paper(doi, file_name, solve_captcha=True)
        if not retrieved:
            logging.critical('PDF ERROR WITH %s (%s)' % (key, item))
            continue

        paperids2data[key] = (file_name, doi, authors, title, true_pubdata, pubyear)
        time.sleep(2)

        if retrieved is True:
            logging.info('PDF OK: %s' % key)
        else:
            logging.critical('MANUAL RETRIEVAL OF %s.pdf: %s' % (key, retrieved))

    logging.info('RESULTS LENGTH: %s' % len(paperids2data))

    f = open(data_step_two, 'w')
    f.write(json.dumps(paperids2data, indent=4))
    f.close()
