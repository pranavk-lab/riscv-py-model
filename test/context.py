#!/usr/bin/env python3
import os
import sys
x = os.path.abspath(os.path.join(os.path.dirname(__file__), '../model'))

sys.path.append(x)

print(x)

from cpu import RV32ICORE
from bit_manipulation import BitManip, XLen
