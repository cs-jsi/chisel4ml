import re
import os
import json
import subprocess
from importlib.metadata import version

# The current directory
__here__ = os.path.dirname(os.path.realpath(__file__))

def git_describe_to_pep440(git_describe):
    parsed = re.findall(r"(\d+\.\d+\.\d+)-(\d+)-(\w+)", git_describe)
    pep440_version = f"{parsed[0][0]}.dev{parsed[0][1]}+{parsed[0][2]}"
    return pep440_version

def installation_is_editable():
    '''
        Returns true if the package was installed with the editable flag.
    '''
    file_exists = os.path.isfile(os.path.join(__here__, '.editable'))
    return file_exists

if installation_is_editable():
    cwd = os.getcwd()
    os.chdir(__here__)
    res = subprocess.run(["git", "describe", "--tags"], capture_output=True)
    os.chdir(cwd)
    encoding = json.detect_encoding(res.stdout)
    git_describe = res.stdout.decode(encoding).rstrip("\r*\n")
    __version__ = git_describe_to_pep440(git_describe)
else:
    __version__ = version("chisel4ml")
