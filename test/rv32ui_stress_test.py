#!/usr/bin/env python3

import sys
sys.path.append('../')
from cpu import RV32ICORE
from bit_manipulation import BitManip32 

TEST_2_ADD = [
    "00002097",
    "e8c08093",
    "0aa00113",
    "00209023",
    "00009703",
    "0aa00393",
    "00200193",
    "44771e63"
    # "44770e63"
]

core = RV32ICORE(8192, "little")
bm = BitManip32()

# Setup code in  memeory
addr = 0x0000174
core.PC = addr
for x in TEST_2_ADD:
    core.memory.write_mem_32(addr, bm.hex_str_2_unsigned_int(x))
    addr+=4

core.core_dump("input_mem.dump", "input_reg.dump")

# Run test code

for x in TEST_2_ADD:
    core.st_run()
    core.core_dump("output_mem.dump", "output_reg.dump")
    # input()
