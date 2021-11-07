riscv-py-model
================

About
-----------

This repository hosts a python model to emulate RISC-V 32-bit I instructions. 
It is only intended for research purposes. 

Not to be used for embedded development or RTL verification. 

Use the following repo for formal RISC-V models used in riscv software 
development and HW DV.
https://github.com/riscv-ovpsim/imperas-riscv-tests.git 

Quickstart
----------------

    $ git clone https://github.com/pranavk-lab/riscv-py-model.git
    $ cd riscv-py-model

Run Coverage
----------------

    $ cd model/test/
    $ chmod +x run_coverage
    $ ./run_coverage

Future Work
----------------

Currently the model only supports 32-bit I instructions. Future plans to 
add following riscv instructions.

TVM Name | Description
--- | ---
`rv32ui` | RV32 user-level, integer only
`rv32si` | RV32 supervisor-level, integer only
`rv64ui` | RV64 user-level, integer only
`rv64uf` | RV64 user-level, integer and floating-point
`rv64uv` | RV64 user-level, integer, floating-point, and vector
`rv64si` | RV64 supervisor-level, integer only
`rv64sv` | RV64 supervisor-level, integer and vector

Source, https://github.com/riscv/riscv-tests

Additionally, it would be nice to parse instructions written in Sail. 
Source, https://github.com/riscv/sail-riscv.git