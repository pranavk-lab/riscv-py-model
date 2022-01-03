riscv-py-model
================

About
-----------

This repository hosts a python model of RISC-V 32-bit I-instruction. 
It is intended for research purposes only. Not to be used for embedded 
development or RTL verification. 

Use https://github.com/riscv-ovpsim/imperas-riscv-tests.git as formal RISC-V 
model for software development and HW DV.

Supported Instructions
--------------------------

TVM Name | Description
--- | ---
`rv32ui` | RV32 user-level, integer only
`rv32si` | RV32 supervisor-level, integer only
`rv64ui` | RV64 user-level, integer only
`rv64si` | RV64 supervisor-level, integer only

Quickstart
----------------

    $ git clone https://github.com/pranavk-lab/riscv-py-model.git
    $ cd riscv-py-model
    $ pip3 install -r requirements.txt

How to Run Model
-------------------------------

Use this example to run a riscv test.

    $ cd examples/
    $ ./add.py

Basic Structure of Test Code
--------------------------------

```python
from cpu import RV32ICORE
from bit_manipulation import BitManip32 

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
core = RV32ICORE(8192, "little")

# Create the bit manipulation instance. (Optional)
bm = BitManip32()

# Copy code in virtual memeory
addr = 0x0000174
core.PC = addr
for x in TEST_2_ADD:
    core.memory.write_mem_32(addr, bm.hex_str_2_unsigned_int(x))
    addr+=4

# Get debug statements. 
core.core_dump("input_mem.dump", "input_reg.dump")

# Run test code
for x in TEST_2_ADD:
    core.st_run()
    core.core_dump("output_mem.dump", "output_reg.dump")
    # input()
```

Future Work
----------------

Currently the model only supports integer instructions. Future work to add
floating point and vector support.

TVM Name | Description
--- | ---
`rv64uf` | RV64 user-level, integer and floating-point
`rv64uv` | RV64 user-level, integer, floating-point, and vector
`rv64sv` | RV64 supervisor-level, integer and vector

Source, https://github.com/riscv/riscv-tests

Additionally, it would be nice to parse instructions written in Sail. 
Source, https://github.com/riscv/sail-riscv.git