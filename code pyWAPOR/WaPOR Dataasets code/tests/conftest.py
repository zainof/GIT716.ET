import os
from datetime import datetime
import pytest
import pywapor
from sys import platform

@pytest.hookimpl(tryfirst=True)
def pytest_configure(config):

    folder = os.path.join(os.path.split(pywapor.__path__[0])[0], "tests", "results")
    
    if not os.path.exists(folder):
        os.makedirs(folder)

    fp = os.path.join(folder, f"{datetime.now():%Y%m%d_%H%M%S}.html")

    config.option.htmlpath = fp

    if platform == "darwin":
        # OS X
        config.option.basetemp = r"/Users/hmcoerver/PyTests"
    elif platform == "win32":
        # Windows
        config.option.basetemp = r"C:\local_data\PyTests"
    