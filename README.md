riscv-py-model
================

About
-----------

This repository hosts a python prototype I personally use to understand 
the inner workings of RISC-V processors. This will not be used for formal 
HW development verification. 

Furthur research into formal HW verification is required using
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

Currently the model only supports 32 bit integer instructions. Future plans to 
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