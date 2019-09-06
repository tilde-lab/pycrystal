
import os
import time
import logging
import json
import re
from unicodedata import normalize

try: from HTMLParser import HTMLParser
except ImportError: from html.parser import HTMLParser

try: from urllib import urlencode
except ImportError: from urllib.parse import urlencode

import httplib2
from pylev import levenshtein


req = httplib2.Http()
h = HTMLParser()
logging.basicConfig(level=logging.INFO)

STOPWORDS = ['sub', 'sup', 'alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta', 'theta', 'iota', 'kappa', 'lambda', 'omicron', 'sigma', 'upsilon', 'omega', 'minus', 'plus', 'degc']

def clean_chars(name):
    return normalize('NFKD', h.unescape(name)).encode('ascii', 'ignore').decode('utf-8')

def format_name(name):
    if '.' not in name:
        return name
    return name.split('.')[-1].strip()

def mine_doi(authors, title, pubyear, pubdata):

    broken = False
    sought_doi = None

    authors = [format_name(clean_chars(x.strip("."))) for x in authors.replace(" and ", ",").split(",")]
    pubyear = int(pubyear)

    failed_attempts = 0

    trial = '%s, "%s", %s, %s' % (", ".join(authors), clean_chars(title), clean_chars(pubdata), pubyear)

    while True:
        try:
            logging.info(trial)
            response, content = req.request('http://search.crossref.org/dois?' + urlencode({'q': trial}), 'GET')
        except:
            logging.error("Provider does not answer")
            if failed_attempts > 7:
                logging.error("NETWORK DOWN")
                broken = True
                break
            time.sleep(3 * failed_attempts)
            failed_attempts += 1
        else:
            failed_attempts = 0
            break

    if broken:
        return False

    try:
        content = json.loads(content.decode('utf-8'))
        res_year = int(content[0]['year'])
        #nsc = int(content[0]['normalizedScore'])
    except:
        logging.error("GOT UNEXPECTED ANSWER: %s" % content)
        return False

    #if res_year == pubyear and nsc == 100:
    if res_year == pubyear:
        sought_doi = content[0]['doi'] #.decode('string-escape')
    else:
        logging.error("YEARS NOT MATCH: %s vs. %s, STOPPING" % (pubyear, res_year))
        return False

    while True:
        try:
            response, content = req.request('http://api.crossref.org/works/' + sought_doi, 'GET')
        except:
            logging.error("Provider does not answer")
            if failed_attempts > 7:
                logging.error("NETWORK DOWN")
                broken = True
                break
            time.sleep(3 * failed_attempts)
            failed_attempts += 1
        else:
            failed_attempts = 0
            break

    if broken:
        return False

    existing_title = re.sub(r"[^a-zA-Z]", "", title.replace(' the ', '').lower())
    for word in STOPWORDS:
        existing_title = existing_title.replace(word, "")
    existing_title = existing_title.replace('i', '')

    existing_authors = sorted(authors)

    try:
        payload = json.loads(content.decode('utf-8'))
    except:
        logging.error("UNEXPECTED ANSWER: %s" % content)
        return False

    found_authors, true_authors = [], []
    try:
        for author in payload['message']['author']:
            found_authors.append( clean_chars(author['family']).capitalize() )
    except KeyError:
        pass
    true_authors = found_authors
    found_authors.sort()

    try:
        found_title = clean_chars(payload['message']['title'][0])
    except:
        logging.critical("EMPTY TITLE")
        return False

    true_title = found_title
    found_title = re.sub(r"[^a-zA-Z]", "", found_title.replace(' the ', '').lower())
    for word in STOPWORDS:
        found_title = found_title.replace(word, "")
    found_title = found_title.replace('i', '')

    if len(found_title) > len(existing_title):
        found_title = found_title[:len(existing_title)]
    elif len(found_title) < len(existing_title):
        existing_title = existing_title[:len(found_title)]

    true_jissue = payload['message'].get('issue', '')
    true_jpage = payload['message'].get('page', '')
    true_pubdata = ''
    journal = payload['message']['container-title'][0]

    if true_jissue and true_jpage:
        true_pubdata = "%s, %s" % (true_jissue, true_jpage)
    elif true_jissue:
        true_pubdata = true_jissue
    elif true_jpage:
        true_pubdata = true_jpage

    true_pubdata = clean_chars(true_pubdata)

    titles_match = (levenshtein(existing_title, found_title) < 5)
    authors_match = False

    for i, j in zip(existing_authors, found_authors):
        authors_match = (levenshtein(i, j) < 3)
        if not authors_match:
            break

    if not titles_match and not authors_match:
        logging.critical("ALL INCORRECT DOI, STOPPING:")
        logging.info("TITLS: %s --- %s" % (existing_title, found_title))
        logging.info("AUTHS: %s --- %s" % (str(existing_authors), str(found_authors)))
        return False

    elif not titles_match:
        logging.critical("UNMATCHING TITLES, STOPPING:\n%s\n%s" % (existing_title, found_title))
        return False

    elif not authors_match:
        logging.info("Unmatching authors:\n%s\n%s" % (existing_authors, found_authors))

    logging.info("TITLS: %s --- %s --- match: %s" % (existing_title, found_title, titles_match))
    logging.info("AUTHS: %s --- %s --- match: %s" % (str(existing_authors), str(found_authors), authors_match))
    logging.info("ORIGN: %s, %s, %s (%s)" % (existing_authors, title, pubdata, pubyear))
    logging.info("FOUND: %s, %s, %s (%s)" % (true_authors, true_title, true_pubdata, pubyear))

    authors = ", ".join(true_authors)
    true_pubdata = journal + '; ' + true_pubdata

    return sought_doi, authors, true_title, true_pubdata, pubyear


if __name__ == "__main__":

    print(mine_doi(
        'Blokhin, Evarestov, Gryaznov, Kotomin, Maier',
        'Theoretical modeling of antiferrodistortive phase transition for STO ultrathin films',
        '2013',
        'Phys Rev B'
    ))
