"""
pip download --only-binary :all: \
    --platform manylinux2014_aarch64 \
    --python-version 3.10 \
    --implementation cp \
    --abi cp310 \
    -d wheelhouse \
    watchdog

pip download --only-binary :all: \
    --platform manylinux2014_aarch64 \
    --python-version 3.10 \
    --implementation cp \
    --abi cp310 \
    -d wheelhouse \
    psutil

python3 -m pip install psutil

"""
# UNZIP FILES TO
# /usr/lib64/python3.10
from math import hypot, sqrt, fabs, exp, erf, tau, log, fsum, sumprod
# /Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10

import math

import pkg_resources
print(pkg_resources.__file__)
import os
