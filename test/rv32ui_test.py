#!/usr/bin/env python3
from context import RV32ICORE
from context import BitManip, XLen
from numpy import uint8, uint16, uint32, int32, iinfo
import unittest


class TestRV32UI(unittest.TestCase):

    REG_FILE = [uint32(0)] * 32

    def init_core(self):
        self.core = RV32ICORE()
    
    def init_memory(self, addr, data):
        self.core.memory.write_mem_32(addr, data)

    def hex_str_2_unsigned_int(self, hex_str: str) -> uint32:
        return BitManip(XLen._32BIT).hex_str_2_unsigned_int(hex_str)

    def run_branch_instructions(self, instr):

        # Create rv32i instance
        self.core = RV32ICORE()

        # Copy register file into core
        self.core.REG_FILE = self.REG_FILE

        self.init_memory(0, self.hex_str_2_unsigned_int(instr))

        # Single test run
        self.core.st_run()

    def test_conditional_branch_equals_true(self):

        # Initialize registers
        self.REG_FILE[1] = -5
        self.REG_FILE[2] = -5

        # Run test
        self.run_branch_instructions("00208663")

        # Compare 
        self.assertEqual(self.core.PC, 6)

    def test_conditional_branch_equals_true(self):

        self.init_core()

        instr = self.hex_str_2_unsigned_int("00208663")
        

        # Initialize self.core registers
        self.core.REG_FILE[1] = -5
        self.core.REG_FILE[2] = -5

        # PC offset
        offset = 6

        # Initialize memory with instruction
        self.init_memory(0, instr)

        # Single test run
        self.core.st_run()

        self.assertEqual(self.core.PC, offset)

    def test_conditional_branch_equals_false(self):

        self.init_core() 
        
        instr = self.hex_str_2_unsigned_int("00208663")

        # Initialize self.core registers
        self.core.REG_FILE[1] = -5
        self.core.REG_FILE[2] = -6

        # PC offset
        offset = 6

        # Initialize memory with instruction
        self.init_memory(0, instr)

        # Single test run
        self.core.st_run()

        self.assertEqual(self.core.PC, 4)

    def test_conditional_branch_not_equals_true(self):

        self.init_core() 
        
        instr = self.hex_str_2_unsigned_int("28301863")

        # Initialize self.core registers
        self.core.REG_FILE[0] = 0
        self.core.REG_FILE[3] = 6

        # PC offset
        offset = 328

        # Initialize memory with instruction
        self.init_memory(0, instr)

        # Single test run
        self.core.st_run()

        self.assertEqual(self.core.PC, offset)

    def test_conditional_branch_not_equals_false(self):

        self.init_core() 
        
        instr = self.hex_str_2_unsigned_int("28301863")

        # Initialize self.core registers
        self.core.REG_FILE[0] = 0
        self.core.REG_FILE[3] = 0

        # PC offset
        offset = 328

        # Initialize memory with instruction
        self.init_memory(0, instr)

        # Single test run
        self.core.st_run()

        self.assertEqual(self.core.PC, 4)

    def test_conditional_branch_lt_true(self):

        self.init_core() 
        
        instr = self.hex_str_2_unsigned_int("00054c63")

        # Initialize self.core registers
        self.core.REG_FILE[10] = -1
        self.core.REG_FILE[0] = 0

        # PC offset
        offset = 12

        # Initialize memory with instruction
        self.init_memory(0, instr)

        # Single test run
        self.core.st_run()

        self.assertEqual(self.core.PC, offset)

    def test_conditional_branch_lt_false(self):

        self.init_core() 
        
        instr = self.hex_str_2_unsigned_int("00054c63")

        # Initialize self.core registers
        self.core.REG_FILE[10] = 1
        self.core.REG_FILE[0] = 0

        # PC offset
        offset = 12

        # Initialize memory with instruction
        self.init_memory(0, instr)

        # Single test run
        self.core.st_run()

        self.assertEqual(self.core.PC, 4)

    def test_conditional_branch_gt_true(self):

        self.init_core() 
        
        instr = self.hex_str_2_unsigned_int("0020d663")

        # Initialize self.core registers
        self.core.REG_FILE[1] = 1
        self.core.REG_FILE[2] = -1

        # PC offset
        offset = 6

        # Initialize memory with instruction
        self.init_memory(0, instr)

        # Single test run
        self.core.st_run()

        self.assertEqual(self.core.PC, offset)

    def test_conditional_branch_gt_false(self):

        self.init_core() 
        
        instr = self.hex_str_2_unsigned_int("0020d663")

        # Initialize self.core registers
        self.core.REG_FILE[1] = -1
        self.core.REG_FILE[2] = 1

        # PC offset
        offset = 6

        # Initialize memory with instruction
        self.init_memory(0, instr)

        # Single test run
        self.core.st_run()

        self.assertEqual(self.core.PC, 4)

    def test_conditional_branch_ltu_true(self):

        self.init_core() 
        
        instr = self.hex_str_2_unsigned_int("0020e663")

        # Initialize self.core registers
        self.core.REG_FILE[1] = uint32(iinfo(uint32).max)-1
        self.core.REG_FILE[2] = uint32(iinfo(uint32).max)

        # PC offset
        offset = 6

        # Initialize memory with instruction
        self.init_memory(0, instr)

        # Single test run
        self.core.st_run()

        self.assertEqual(self.core.PC, offset)

    def test_conditional_branch_ltu_false(self):

        self.init_core() 
        
        instr = self.hex_str_2_unsigned_int("0020e663")

        # Initialize self.core registers
        self.core.REG_FILE[1] = uint32(iinfo(uint32).max)
        self.core.REG_FILE[2] = uint32(iinfo(uint32).max)

        # PC offset
        offset = 6

        # Initialize memory with instruction
        self.init_memory(0, instr)

        # Single test run
        self.core.st_run()

        self.assertEqual(self.core.PC, 4)

    def test_conditional_branch_geu_true(self):

        self.init_core() 
        
        instr = self.hex_str_2_unsigned_int("0020f663")

        # Initialize self.core registers
        self.core.REG_FILE[1] = uint32(iinfo(uint32).max)
        self.core.REG_FILE[2] = uint32(iinfo(uint32).max)-1

        # PC offset
        offset = 6

        # Initialize memory with instruction
        self.init_memory(0, instr)

        # Single test run
        self.core.st_run()

        self.assertEqual(self.core.PC, offset)

    def test_conditional_branch_geu_false(self):

        self.init_core() 
        
        instr = self.hex_str_2_unsigned_int("0020f663")

        # Initialize self.core registers
        self.core.REG_FILE[1] = uint32(iinfo(uint32).max)
        self.core.REG_FILE[2] = uint32(iinfo(uint32).max)

        # PC offset
        offset = 6

        # Initialize memory with instruction
        self.init_memory(0, instr)

        # Single test run
        self.core.st_run()

        self.assertEqual(self.core.PC, 4)

    def test_unknown_opcode1_0_value_error(self):

        self.init_core() 
        
        instr = self.hex_str_2_unsigned_int("0020f662")

        self.assertRaises(ValueError, self.core.decode, instr)

    def test_unknown_opcode6_2_value_error(self):

        self.init_core() 
        
        instr = self.hex_str_2_unsigned_int("0020f6f7")

        self.assertRaises(ValueError, self.core.decode, instr)

    def test_jump_and_link(self):

        self.init_core()
        
        instr = self.hex_str_2_unsigned_int("0100026f")

        PC_test = 0x4 * 20

        # Initialize PC
        self.core.PC = PC_test

        # PC offset
        offset = PC_test + 8

        # Initialize memory with instruction
        self.init_memory(self.core.PC, instr)

        # Single test run
        self.core.st_run()

        self.assertEqual(self.core.PC, offset)

        self.assertEqual(self.core.REG_FILE[4], PC_test + 0x4) 

    def test_jump_and_link_register(self):

        self.init_core()
        
        instr = self.hex_str_2_unsigned_int("000302e7")

        PC_test = 0x4 * 20

        # Initialize PC
        self.core.PC = PC_test

        self.core.REG_FILE[6] = uint32(13)

        # PC offset
        offset = 12

        # Initialize memory with instruction
        self.init_memory(self.core.PC, instr)

        # Single test run
        self.core.st_run()

        self.assertEqual(self.core.PC, offset)

        self.assertEqual(self.core.REG_FILE[5], PC_test + 0x4) 

    def test_load_upper_imm(self):

        self.init_core()
        
        instr = self.hex_str_2_unsigned_int("800002b7")

        PC_test = 0x4 * 20

        # Initialize PC
        self.core.PC = PC_test

        # upper immediate. pre-calculated
        u_imm = 2147483648

        # destination register
        dst = 5

        # Initialize memory with instruction
        self.init_memory(self.core.PC, instr)

        # Single test run
        self.core.st_run()

        self.assertEqual(self.core.PC, PC_test + 0x4)

        self.assertEqual(self.core.REG_FILE[dst], u_imm) 

    def test_add_upper_imm_2_pc(self):

        self.init_core()
        
        instr = self.hex_str_2_unsigned_int("00004517")

        PC_test = 0x4 * 20

        # Initialize PC
        self.core.PC = PC_test

        # upper immediate. pre-calculated
        u_imm =  16384 + PC_test

        # destination register
        dst = 10

        # Initialize memory with instruction
        self.init_memory(self.core.PC, instr)

        # Single test run
        self.core.st_run()

        self.assertEqual(self.core.PC, PC_test + 0x4)

        self.assertEqual(self.core.REG_FILE[dst], u_imm) 

    def test_reg_imm_add_positive(self):

        self.init_core()
        
        instr = self.hex_str_2_unsigned_int("00108713")

        PC_test = 0x4 * 20

        # Initialize PC
        self.core.PC = PC_test

        # Set up src register
        src_val = 5
        self.core.REG_FILE[1] = src_val

        # upper immediate. pre-calculated
        imm =  1

        # destination register
        dst = 14

        # Initialize memory with instruction
        self.init_memory(self.core.PC, instr)

        # Single test run
        self.core.st_run()

        self.assertEqual(self.core.PC, PC_test + 0x4)

        self.assertEqual(int32(self.core.REG_FILE[dst]), imm + src_val) 

    def test_reg_imm_add_negetive(self):

        self.init_core()
        
        instr = self.hex_str_2_unsigned_int("00108713")

        PC_test = 0x4 * 20

        # Initialize PC
        self.core.PC = PC_test

        # Set up src register
        src_val = -5
        self.core.REG_FILE[1] = src_val

        # upper immediate. pre-calculated
        imm =  1

        # destination register
        dst = 14

        # Initialize memory with instruction
        self.init_memory(self.core.PC, instr)

        # Single test run
        self.core.st_run()

        self.assertEqual(self.core.PC, PC_test + 0x4)

        self.assertEqual(int32(self.core.REG_FILE[dst]), imm + src_val) 

    def test_reg_imm_shift_left(self):

        self.init_core()
        
        instr = self.hex_str_2_unsigned_int("00709713")

        PC_test = 0x4 * 20

        # Initialize PC
        self.core.PC = PC_test

        # Set up src register
        src_val = uint32(iinfo(uint32).max)
        self.core.REG_FILE[1] = src_val

        # upper immediate. pre-calculated
        imm =  1

        # destination register
        dst = 14

        # Initialize memory with instruction
        self.init_memory(self.core.PC, instr)

        # Single test run
        self.core.st_run()

        self.assertEqual(self.core.PC, PC_test + 0x4)

        self.assertEqual(self.core.REG_FILE[dst], 0xffffff80) 

    def test_reg_imm_shift_right_logic(self):

        self.init_core()
        
        instr = self.hex_str_2_unsigned_int("0070d713")

        PC_test = 0x4 * 20

        # Initialize PC
        self.core.PC = PC_test

        # Set up src register
        src_val = uint32(iinfo(uint32).max)
        self.core.REG_FILE[1] = src_val

        # upper immediate. pre-calculated
        imm =  1

        # destination register
        dst = 14

        # Initialize memory with instruction
        self.init_memory(self.core.PC, instr)

        # Single test run
        self.core.st_run()

        self.assertEqual(self.core.PC, PC_test + 0x4)

        self.assertEqual(self.core.REG_FILE[dst], 0x01ffffff) 

    def test_reg_imm_shift_right_arith(self):
        self.init_core()
        
        instr = self.hex_str_2_unsigned_int("4070d713")

        PC_test = 0x4 * 20

        # Initialize PC
        self.core.PC = PC_test

        # Set up src register
        src_val = 0xf0ffffff
        self.core.REG_FILE[1] = src_val

        # upper immediate. pre-calculated
        imm =  1

        # destination register
        dst = 14

        # Initialize memory with instruction
        self.init_memory(self.core.PC, instr)

        # Single test run
        self.core.st_run()

        self.assertEqual(self.core.PC, PC_test + 0x4)

        self.assertEqual(self.core.REG_FILE[dst], 0xffe1ffff) 

    def test_shift_imm11_5_exception(self):

        self.init_core() 
        
        instr = self.hex_str_2_unsigned_int("5070d713")

        self.assertRaises(ValueError, self.core.execute, self.core.decode(instr))

    def test_reg_imm_slti_true(self):
        self.init_core()
        
        instr = self.hex_str_2_unsigned_int("0070a713")

        PC_test = 0x4 * 20

        # Initialize PC
        self.core.PC = PC_test

        # Set up src register
        src_val = -5
        self.core.REG_FILE[1] = src_val

        # upper immediate. pre-calculated
        imm =  1

        # destination register
        dst = 14

        # Initialize memory with instruction
        self.init_memory(self.core.PC, instr)

        # Single test run
        self.core.st_run()

        self.assertEqual(self.core.PC, PC_test + 0x4)

        self.assertEqual(self.core.REG_FILE[dst], 0x1) 

    def test_reg_imm_slti_false(self):
        self.init_core()
        
        instr = self.hex_str_2_unsigned_int("8000a713")

        PC_test = 0x4 * 20

        # Initialize PC
        self.core.PC = PC_test

        # Set up src register
        src_val = -5
        self.core.REG_FILE[1] = src_val

        # upper immediate. pre-calculated
        imm =  1

        # destination register
        dst = 14

        # Initialize memory with instruction
        self.init_memory(self.core.PC, instr)

        # Single test run
        self.core.st_run()

        self.assertEqual(self.core.PC, PC_test + 0x4)

        self.assertEqual(self.core.REG_FILE[dst], 0x0) 

    def test_reg_imm_sltiu_true(self):
        self.init_core()
        
        instr = self.hex_str_2_unsigned_int("8000b713")

        PC_test = 0x4 * 20

        # Initialize PC
        self.core.PC = PC_test

        # Set up src register
        src_val = 7
        self.core.REG_FILE[1] = src_val

        # upper immediate. pre-calculated
        imm =  1

        # destination register
        dst = 14

        # Initialize memory with instruction
        self.init_memory(self.core.PC, instr)

        # Single test run
        self.core.st_run()

        self.assertEqual(self.core.PC, PC_test + 0x4)

        self.assertEqual(self.core.REG_FILE[dst], 0x1) 

    def test_reg_imm_sltiu_false(self):
        self.init_core()
        
        instr = self.hex_str_2_unsigned_int("0070b713")

        PC_test = 0x4 * 20

        # Initialize PC
        self.core.PC = PC_test

        # Set up src register
        src_val = -5
        self.core.REG_FILE[1] = src_val

        # upper immediate. pre-calculated
        imm =  1

        # destination register
        dst = 14

        # Initialize memory with instruction
        self.init_memory(self.core.PC, instr)

        # Single test run
        self.core.st_run()

        self.assertEqual(self.core.PC, PC_test + 0x4)

        self.assertEqual(self.core.REG_FILE[dst], 0x0) 

    def test_reg_imm_xor(self):
        self.init_core()
        
        instr = self.hex_str_2_unsigned_int("f0f0c713")

        PC_test = 0x4 * 20

        # Initialize PC
        self.core.PC = PC_test

        # Set up src register
        src_val = -5
        self.core.REG_FILE[1] = src_val

        # upper immediate. pre-calculated
        imm =  1

        # destination register
        dst = 14

        # Initialize memory with instruction
        self.init_memory(self.core.PC, instr)

        # Single test run
        self.core.st_run()

        self.assertEqual(self.core.PC, PC_test + 0x4)

        self.assertEqual(self.core.REG_FILE[dst], 0xf4) 

    def test_reg_imm_or(self):
        self.init_core()
        
        instr = self.hex_str_2_unsigned_int("f0f0e713")

        PC_test = 0x4 * 20

        # Initialize PC
        self.core.PC = PC_test

        # Set up src register
        src_val = -5
        self.core.REG_FILE[1] = src_val

        # upper immediate. pre-calculated
        imm =  1

        # destination register
        dst = 14

        # Initialize memory with instruction
        self.init_memory(self.core.PC, instr)

        # Single test run
        self.core.st_run()

        self.assertEqual(self.core.PC, PC_test + 0x4)

        self.assertEqual(self.core.REG_FILE[dst], 0xffffffff) 

    def test_reg_imm_and(self):
        self.init_core()
        
        instr = self.hex_str_2_unsigned_int("f0f0f713")

        PC_test = 0x4 * 20

        # Initialize PC
        self.core.PC = PC_test

        # Set up src register
        src_val = -5
        self.core.REG_FILE[1] = src_val

        # upper immediate. pre-calculated
        imm =  1

        # destination register
        dst = 14

        # Initialize memory with instruction
        self.init_memory(self.core.PC, instr)

        # Single test run
        self.core.st_run()

        self.assertEqual(self.core.PC, PC_test + 0x4)

        self.assertEqual(self.core.REG_FILE[dst], 0xffffff0b) 

    def test_reg_reg_add_positive(self):

        self.init_core()
        
        instr = self.hex_str_2_unsigned_int("00208733")

        PC_test = 0x4 * 20

        # Initialize PC
        self.core.PC = PC_test

        # Set up src register
        src1_val = 5
        self.core.REG_FILE[1] = src1_val

        src2_val = 4
        self.core.REG_FILE[2] = src2_val

        # destination register
        dst = 14

        # Initialize memory with instruction
        self.init_memory(self.core.PC, instr)

        # Single test run
        self.core.st_run()

        self.assertEqual(self.core.PC, PC_test + 0x4)

        self.assertEqual(int32(self.core.REG_FILE[dst]), src1_val + src2_val) 

    def test_reg_reg_add_negetive(self):

        self.init_core()
        
        instr = self.hex_str_2_unsigned_int("00208733")

        PC_test = 0x4 * 20

        # Initialize PC
        self.core.PC = PC_test

        # Set up src register
        src1_val = -5
        self.core.REG_FILE[1] = src1_val

        src2_val = -4
        self.core.REG_FILE[2] = src2_val

        # destination register
        dst = 14

        # Initialize memory with instruction
        self.init_memory(self.core.PC, instr)

        # Single test run
        self.core.st_run()

        self.assertEqual(self.core.PC, PC_test + 0x4)

        self.assertEqual(int32(self.core.REG_FILE[dst]), -9) 

    def test_reg_reg_sub_positive(self):

        self.init_core()
        
        instr = self.hex_str_2_unsigned_int("40208733")

        PC_test = 0x4 * 20

        # Initialize PC
        self.core.PC = PC_test

        # Set up src register
        src1_val = 5
        self.core.REG_FILE[1] = src1_val

        src2_val = 4
        self.core.REG_FILE[2] = src2_val

        # destination register
        dst = 14

        # Initialize memory with instruction
        self.init_memory(self.core.PC, instr)

        # Single test run
        self.core.st_run()

        self.assertEqual(self.core.PC, PC_test + 0x4)

        self.assertEqual(int32(self.core.REG_FILE[dst]), src1_val - src2_val) 

    def test_reg_reg_add_negetive(self):

        self.init_core()
        
        instr = self.hex_str_2_unsigned_int("40208733")

        PC_test = 0x4 * 20

        # Initialize PC
        self.core.PC = PC_test

        # Set up src register
        src1_val = -5
        self.core.REG_FILE[1] = src1_val

        src2_val = -4
        self.core.REG_FILE[2] = src2_val

        # destination register
        dst = 14

        # Initialize memory with instruction
        self.init_memory(self.core.PC, instr)

        # Single test run
        self.core.st_run()

        self.assertEqual(self.core.PC, PC_test + 0x4)

        self.assertEqual(int32(self.core.REG_FILE[dst]), -1) 

    def test_shift_funct7_exception_reg_reg_int(self):

        self.init_core() 
        
        instr = self.hex_str_2_unsigned_int("50208733")

        self.assertRaises(ValueError, self.core.execute, self.core.decode(instr))

    def test_reg_reg_shift_left(self):

        self.init_core()
        
        instr = self.hex_str_2_unsigned_int("00209733")

        PC_test = 0x4 * 20

        # Initialize PC
        self.core.PC = PC_test

        # Set up src register
        src1_val = 1
        self.core.REG_FILE[1] = src1_val

        src2_val = 4
        self.core.REG_FILE[2] = src2_val

        # destination register
        dst = 14

        # Initialize memory with instruction
        self.init_memory(self.core.PC, instr)

        # Single test run
        self.core.st_run()

        self.assertEqual(self.core.PC, PC_test + 0x4)

        self.assertEqual(self.core.REG_FILE[dst], 0x10) 

    def test_reg_reg_shift_right_logic(self):

        self.init_core()
        
        instr = self.hex_str_2_unsigned_int("0020d733")

        PC_test = 0x4 * 20

        # Initialize PC
        self.core.PC = PC_test

        # Set up src register
        src1_val = uint32(iinfo(uint32).max)
        self.core.REG_FILE[1] = src1_val

        src2_val = 4
        self.core.REG_FILE[2] = src2_val

        # destination register
        dst = 14

        # Initialize memory with instruction
        self.init_memory(self.core.PC, instr)

        # Single test run
        self.core.st_run()

        self.assertEqual(self.core.PC, PC_test + 0x4)

        self.assertEqual(self.core.REG_FILE[dst], 0x0fffffff) 

    def test_reg_reg_shift_right_arith(self):
        self.init_core()
        
        instr = self.hex_str_2_unsigned_int("4020d733")

        PC_test = 0x4 * 20

        # Initialize PC
        self.core.PC = PC_test

        # Set up src register
        src1_val = 0xf0ffffff
        self.core.REG_FILE[1] = src1_val

        src2_val = 7
        self.core.REG_FILE[2] = src2_val
        # Set up src register

        # destination register
        dst = 14

        # Initialize memory with instruction
        self.init_memory(self.core.PC, instr)

        # Single test run
        self.core.st_run()

        self.assertEqual(self.core.PC, PC_test + 0x4)

        self.assertEqual(self.core.REG_FILE[dst], 0xffe1ffff) 

    def test_shift_imm11_5_exception_reg_reg_int(self):

        self.init_core() 
        
        instr = self.hex_str_2_unsigned_int("5070d733")

        self.assertRaises(ValueError, self.core.execute, self.core.decode(instr))

    def test_reg_reg_slti_true(self):
        self.init_core()
        
        instr = self.hex_str_2_unsigned_int("0020a733")

        PC_test = 0x4 * 20

        # Initialize PC
        self.core.PC = PC_test

        # Set up src register
        src1_val = -5
        self.core.REG_FILE[1] = src1_val

        src2_val = 7
        self.core.REG_FILE[2] = src2_val

        # destination register
        dst = 14

        # Initialize memory with instruction
        self.init_memory(self.core.PC, instr)

        # Single test run
        self.core.st_run()

        self.assertEqual(self.core.PC, PC_test + 0x4)

        self.assertEqual(self.core.REG_FILE[dst], 0x1) 

    def test_reg_reg_slti_false(self):
        self.init_core()
        
        instr = self.hex_str_2_unsigned_int("0020a733")

        PC_test = 0x4 * 20

        # Initialize PC
        self.core.PC = PC_test

        # Set up src register
        src1_val = 7
        self.core.REG_FILE[1] = src1_val

        src2_val = -5
        self.core.REG_FILE[2] = src2_val

        # destination register
        dst = 14

        # Initialize memory with instruction
        self.init_memory(self.core.PC, instr)

        # Single test run
        self.core.st_run()

        self.assertEqual(self.core.PC, PC_test + 0x4)

        self.assertEqual(self.core.REG_FILE[dst], 0x0) 

    def test_reg_reg_sltiu_true(self):
        self.init_core()
        
        instr = self.hex_str_2_unsigned_int("0020b733")

        PC_test = 0x4 * 20

        # Initialize PC
        self.core.PC = PC_test

        # Set up src register
        src1_val = 7
        self.core.REG_FILE[1] = src1_val

        src2_val = -5
        self.core.REG_FILE[2] = src2_val

        # destination register
        dst = 14

        # Initialize memory with instruction
        self.init_memory(self.core.PC, instr)

        # Single test run
        self.core.st_run()

        self.assertEqual(self.core.PC, PC_test + 0x4)

        self.assertEqual(self.core.REG_FILE[dst], 0x1) 

    def test_reg_reg_sltiu_false(self):
        self.init_core()
        
        instr = self.hex_str_2_unsigned_int("0070b733")

        PC_test = 0x4 * 20

        # Initialize PC
        self.core.PC = PC_test

        # Set up src register
        src1_val = -5
        self.core.REG_FILE[1] = src1_val

        src2_val = 7
        self.core.REG_FILE[2] = src2_val

        # destination register
        dst = 14

        # Initialize memory with instruction
        self.init_memory(self.core.PC, instr)

        # Single test run
        self.core.st_run()

        self.assertEqual(self.core.PC, PC_test + 0x4)

        self.assertEqual(self.core.REG_FILE[dst], 0x0) 

    def test_reg_reg_xor(self):
        self.init_core()
        
        instr = self.hex_str_2_unsigned_int("0020c733")

        PC_test = 0x4 * 20

        # Initialize PC
        self.core.PC = PC_test

        # Set up src register
        src1_val = -5
        self.core.REG_FILE[1] = src1_val

        src2_val = -241
        self.core.REG_FILE[2] = src2_val

        # destination register
        dst = 14

        # Initialize memory with instruction
        self.init_memory(self.core.PC, instr)

        # Single test run
        self.core.st_run()

        self.assertEqual(self.core.PC, PC_test + 0x4)

        self.assertEqual(self.core.REG_FILE[dst], 0xf4) 

    def test_reg_reg_or(self):
        self.init_core()
        
        instr = self.hex_str_2_unsigned_int("0020e733")

        PC_test = 0x4 * 20

        # Initialize PC
        self.core.PC = PC_test

        # Set up src register
        src1_val = -5
        self.core.REG_FILE[1] = src1_val

        src2_val = -241
        self.core.REG_FILE[2] = src2_val

        # destination register
        dst = 14

        # Initialize memory with instruction
        self.init_memory(self.core.PC, instr)

        # Single test run
        self.core.st_run()

        self.assertEqual(self.core.PC, PC_test + 0x4)

        self.assertEqual(self.core.REG_FILE[dst], 0xffffffff) 

    def test_reg_reg_and(self):
        self.init_core()
        
        instr = self.hex_str_2_unsigned_int("0020f733")

        PC_test = 0x4 * 20

        # Initialize PC
        self.core.PC = PC_test

        # Set up src register
        src1_val = -5
        self.core.REG_FILE[1] = src1_val

        src2_val = -241
        self.core.REG_FILE[2] = src2_val

        # destination register
        dst = 14

        # Initialize memory with instruction
        self.init_memory(self.core.PC, instr)

        # Single test run
        self.core.st_run()

        self.assertEqual(self.core.PC, PC_test + 0x4)

        self.assertEqual(self.core.REG_FILE[dst], 0xffffff0b) 

    def test_load_byte(self):
        self.init_core()
        
        instr = self.hex_str_2_unsigned_int("ffd08703")

        # Test values
        src = 1
        dest = 14
        imm = -3

        self.core.REG_FILE[src] = 0x25
        self.core.memory.write_mem_32(0x20, 0x53a46732)

        # Initialize PC
        PC_test = 0x4 * 20
        self.core.PC = PC_test

        # Initialize memory with instruction
        self.init_memory(self.core.PC, instr)

        # Single test run
        self.core.st_run()

        self.assertEqual(uint8(self.core.REG_FILE[dest]), 0xa4)

    def test_load_16_bits(self):
        self.init_core()
        
        instr = self.hex_str_2_unsigned_int("ffa09703")

        # Test values
        src = 1
        dest = 14
        imm = -6

        self.core.REG_FILE[src] = 0x25
        self.core.memory.write_mem_32(0x1d, 0x53a46732)

        # Initialize PC
        PC_test = 0x4 * 20
        self.core.PC = PC_test

        # Initialize memory with instruction
        self.init_memory(self.core.PC, instr)

        # Single test run
        self.core.st_run()

        self.assertEqual(uint16(self.core.REG_FILE[dest]), 0x53a4)

    def test_load_word(self):
        self.init_core()
        
        instr = self.hex_str_2_unsigned_int("ff40a703")

        # Test values
        src = 1
        dest = 14
        imm = -12

        self.core.REG_FILE[src] = 0x20
        self.core.memory.write_mem_32(0x14, 0x53a46732)

        # Initialize PC
        PC_test = 0x4 * 20
        self.core.PC = PC_test

        # Initialize memory with instruction
        self.init_memory(self.core.PC, instr)

        # Single test run
        self.core.st_run()

        self.assertEqual((self.core.REG_FILE[dest]), 0x53a46732)

    def test_store_byte(self):
        self.init_core()
        
        instr = self.hex_str_2_unsigned_int("fe208ea3")

        # Test values
        src1 = 1
        src2 = 2
        imm = -3

        self.core.REG_FILE[src1] = 0x23
        self.core.REG_FILE[src2] = 0x53a46732

        # Initialize PC
        PC_test = 0x4 * 20
        self.core.PC = PC_test

        # Initialize memory with instruction
        self.init_memory(self.core.PC, instr)

        # Single test run
        self.core.st_run()

        self.assertEqual(self.core.memory.read_mem_8(0x20), 0x32)

    def test_store_16_bits(self):
        self.init_core()
        
        instr = self.hex_str_2_unsigned_int("fe209d23")

        # Test values
        src1 = 1
        src2 = 2
        imm = -6

        self.core.REG_FILE[src1] = 0x26
        self.core.REG_FILE[src2] = 0x53a46732

        # Initialize PC
        PC_test = 0x4 * 20
        self.core.PC = PC_test

        # Initialize memory with instruction
        self.init_memory(self.core.PC, instr)

        # Single test run
        self.core.st_run()

        self.assertEqual(self.core.memory.read_mem_16(0x20), 0x6732)

    def test_store_word(self):
        self.init_core()
        
        instr = self.hex_str_2_unsigned_int("fe20aa23")

        # Test values
        src1 = 1
        src2 = 2
        imm = -12

        self.core.REG_FILE[src1] = 0x2c
        self.core.REG_FILE[src2] = 0x53a46732

        # Initialize PC
        PC_test = 0x4 * 20
        self.core.PC = PC_test

        # Initialize memory with instruction
        self.init_memory(self.core.PC, instr)

        # Single test run
        self.core.st_run()

        self.assertEqual(self.core.memory.read_mem_32(0x20), 0x53a46732)

unittest.main()