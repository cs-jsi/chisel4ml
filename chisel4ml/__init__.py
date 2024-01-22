import subprocess
import os
from importlib.metadata import version
import json

# The current directory
__here__ = os.path.dirname(os.path.realpath(__file__))

def installation_is_editable():
    '''
        Returns true if the package was installed with the editable flag.
    '''
    not_to_install_exists = os.path.isfile(os.path.join(__here__, '.editable'))
    return not_to_install_exists

if installation_is_editable():
    res = subprocess.run(["git", "describe", "--tags"], capture_output=True)
    encoding = json.detect_encoding(res.stdout)
    __version__ = res.stdout.decode(encoding).rstrip("\r*\n")
else:
    __version__ = version("chisel4ml")
