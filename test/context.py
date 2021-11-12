#!/usr/bin/env python3
import os
import sys
x = os.path.abspath(os.path.join(os.path.dirname(__file__), '../model'))

sys.path.append(x)

print(x)

import cpu
from rv32ui import RV32UI
from rv64ui import RV64UI
from bit_manipulation import BitManip, XLen
