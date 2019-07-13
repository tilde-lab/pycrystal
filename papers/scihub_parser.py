
from __future__ import division
import os
import re
import logging
import random

from bs4 import BeautifulSoup as bs
import requests
from python3_anticaptcha import ImageToTextTask


ANTICAPTCHA_KEY = ""
logging.basicConfig(level=logging.INFO)


def retrieve_paper(doi, file_name, solve_captcha=False):

    target = 'http://sci-hub.tw/' + doi

    response = requests.get(target)
    soup = bs(response.content, "html.parser")

    try:
        mirror = soup.find("iframe", attrs={"id": "pdf"})['src'].split("#")[0]
        if mirror.startswith('//'):
            mirror = mirror[2:]
            mirror = 'https://' + mirror
    except Exception:
        logging.error("Mirror not found")
        return False

    try:
        doi = soup.title.text.split("|")[2].strip()
    except Exception:
        logging.error("DOI not found")
        return False

    logging.info("Get %s from %s" % (doi, mirror))

    response = requests.get(mirror)

    if response.headers['content-type'] == "application/pdf":

        logging.info("Downloaded %2.2f MB\n" % (int(response.headers['Content-Length']) / 1000000))

        with open(file_name, "wb") as f:
            f.write(response.content)
        f.close()
        return True

    elif re.match("text/html", response.headers['content-type']):
        logging.info("Looks like captcha encountered")
        logging.info("Download link is \n" + mirror + "\n")

        if not solve_captcha:
            return mirror

        captcha = bs(response.content, "html.parser")
        try:
            img_url = captcha.find("img", attrs={"id": "captcha"})['src']
        except Exception:
            logging.error("CAPTCHA SOLVING: Cannot get captcha image")
            return False

        img_url = 'http://' + mirror[7:].split('/', 2)[0] + img_url
        logging.info(img_url)

        r = requests.get(img_url)
        f = open('/tmp/captcha.jpg', 'wb')
        f.write(r.content)
        f.close()

        user_answer = ImageToTextTask.ImageToTextTask(anticaptcha_key=ANTICAPTCHA_KEY).captcha_handler(
            captcha_file='/tmp/captcha.jpg'
        ) # NB *captcha_link* doesn't always work
        try:
            solution = user_answer['solution']['text']
        except KeyError:
            logging.error('CAPTCHA SOLVING: Unexpected anticaptcha response: %s' % user_answer)
            return False

        img_id = img_url.split('/')[-1].replace('.jpg', '')
        response = requests.post(mirror, data={'answer': solution, 'id': img_id})

        if response.status_code != 200:
            logging.error('CAPTCHA SOLVING: Anticaptcha gave wrong answer, error %s' % response.status_code)
            return False

        if response.headers['content-type'] != "application/pdf":
            logging.error('CAPTCHA SOLVING: Redirection failed, answer is %s' % response.headers['content-type'])
            return False

        logging.info("Downloaded %2.2f MB\n" % (int(response.headers['Content-Length']) / 1000000))

        with open(file_name, "wb") as f:
            f.write(response.content)
        f.close()
        return True

if __name__ == "__main__":

    print(retrieve_paper('10.1103/physrevb.88.241407', 'blokhin2013.pdf', solve_captcha=True))
