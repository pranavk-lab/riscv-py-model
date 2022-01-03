#!/usr/bin/env python3

import sys
sys.path.append('../model/')
from cpu import RISCVCore
from rv32ui import RV32UI
from bit_manipulation import XLen

# Specify the test hex codes 
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

# Create the riscv instance. 
core = RISCVCore(isa=RV32UI(), xlen=XLen._32BIT)

# Copy code in virtual memeory
addr = 0x0000174
core.PC = addr
for x in TEST_2_ADD:
    core.memory.write_mem_32(addr, core.bm.hex_str_2_unsigned_int(x))
    addr+=4

# Get debug statements. 
core.core_dump("input_mem.dump", "input_reg.dump")

# Run test code
for x in TEST_2_ADD:
    core.st_run()
    core.core_dump("output_mem.dump", "output_reg.dump")
    # input()
