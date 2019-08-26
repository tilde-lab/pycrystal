Articles related to CRYSTAL code
==========

This is a set of scripts to parse and systematically download
the bibliographic information, presented at http://www.crystal.unito.it

Deps:
```
pip install requests bs4 pylev python3_anticaptcha
```

Check anti-captcha balance:
```
#!/usr/bin/env python3
from python3_anticaptcha import AntiCaptchaControl; print(AntiCaptchaControl.AntiCaptchaControl(anticaptcha_key="").get_balance())
```
