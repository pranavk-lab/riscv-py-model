#!/usr/bin/env python3
from logging import setLogRecordFactory
import sys
sys.path.append('../')
from bit_manipulation import BitManip32 
from cpu import RV32ICORE, RegImmInt_32, RegRegInt_32
from numpy import binary_repr, isfinite, issubdtype, uint32, int32, iinfo
import unittest
import coverage


class TestBitManip32(unittest.TestCase):

    hex_str_high_msb = "f7830000"
    hex_str_low_msb = "77830521"

    def test_hex_str_2_unsigned_int(self):
        bm = BitManip32()
        hex_str = self.hex_str_high_msb
        vector = bm.hex_str_2_unsigned_int(hex_str)
        self.assertEqual(issubdtype(vector, uint32), True)
        
    def test_get_sub_bits_from_instr(self):
        # Test get_sub_bits_from_instr 
        upper = 31
        lower = 12
        bm = BitManip32()
        hex_str = self.hex_str_high_msb
        vector = bm.hex_str_2_unsigned_int(hex_str)
        vector_unsigned, width = bm.get_sub_bits_from_instr(vector, upper, lower)
        self.assertEqual(vector_unsigned, 1013808)
        vector_unsigned_binary = binary_repr(vector_unsigned, width=width)
        self.assertEqual(vector_unsigned_binary, '11110111100000110000')
        self.assertRaises(ValueError, bm.hex_str_2_unsigned_int, 'f' + self.hex_str_low_msb)
        self.assertEqual(width, (upper-lower) + 1)

    def test_sub_bits_from_instr_negetive_number(self):
        # Test get_sub_bits_from_instr negetive number
        upper = 31
        lower = 12
        bm = BitManip32()
        hex_str = self.hex_str_high_msb
        vector = bm.hex_str_2_unsigned_int(hex_str)
        vector_signed = bm.sign_extend_nbit_2_int32(bm.get_sub_bits_from_instr(vector, upper, lower))
        self.assertEqual(vector_signed, -34768)
        vector_signed_binary = binary_repr(vector_signed, width=32)
        self.assertEqual(vector_signed_binary, '11111111111111110111100000110000')
        self.assertRaises(ValueError, bm.get_sub_bits_from_instr, vector, lower, upper)

    def test_get_sub_bits_from_instr_positive_number(self):
        # Test get_sub_bits_from_instr positive number
        upper = 31
        lower = 12
        bm = BitManip32()
        hex_str = self.hex_str_high_msb
        vector = bm.hex_str_2_unsigned_int(hex_str)
        vector_signed = bm.sign_extend_nbit_2_uint32(bm.get_sub_bits_from_instr(vector, upper, lower))
        self.assertEqual(vector_signed, 4294932528)
        vector_signed_binary = binary_repr(vector_signed, width=32) 
        self.assertEqual(vector_signed_binary, '11111111111111110111100000110000')

    def test_concat_S_type_imm(self):
        upper = 31
        lower = 25
        bm = BitManip32()
        hex_str = self.hex_str_low_msb
        vector = bm.hex_str_2_unsigned_int(hex_str)
        imm11_5, width11_5 = bm.get_sub_bits_from_instr(vector, upper, lower)
        self.assertEqual(imm11_5, 59)
        binary_imm11_5 = binary_repr(imm11_5, width11_5)
        self.assertEqual(binary_imm11_5, '0111011')
        self.assertEqual(width11_5, (upper-lower) + 1)

        upper = 11
        lower = 7
        vector = bm.hex_str_2_unsigned_int(hex_str)
        imm4_0, width4_0 = bm.get_sub_bits_from_instr(vector, upper, lower)
        self.assertEqual(imm4_0, 10)
        binary_imm4_0 = binary_repr(imm4_0, width4_0)
        self.assertEqual(binary_imm4_0, '01010')
        self.assertEqual(width4_0, (upper-lower) + 1)

        unsigned_concat_bits, width_simm = bm.concat_bits([(imm11_5, width11_5), (imm4_0, width4_0)])
        concat_bits = bm.sign_extend_nbit_2_int32((unsigned_concat_bits, width_simm))
        binary_concat_bits = binary_repr(concat_bits, width_simm)
        self.assertEqual(concat_bits, 1898)
        self.assertEqual(binary_concat_bits, '011101101010')

    def test_concat_J_imm(self):
        upper = 31
        lower = 31
        bm = BitManip32()
        hex_str = self.hex_str_low_msb
        vector = bm.hex_str_2_unsigned_int(hex_str)
        imm20, width20 = bm.get_sub_bits_from_instr(vector, upper, lower)
        self.assertEqual(imm20, 0)
        binary_imm20 = binary_repr(imm20, width20)
        self.assertEqual(binary_imm20, '0')
        self.assertEqual(width20, (upper-lower) + 1)

        upper = 19
        lower = 12 
        imm19_12, width19_12 = bm.get_sub_bits_from_instr(vector, upper, lower)
        self.assertEqual(imm19_12, 48)
        binary_imm19_12 = binary_repr(imm19_12, width19_12)
        self.assertEqual(binary_imm19_12, '00110000')
        self.assertEqual(width19_12, (upper-lower) + 1)

        upper = 20
        lower = 20
        imm11, width11 = bm.get_sub_bits_from_instr(vector, upper, lower)
        self.assertEqual(imm11, 0)
        binary_imm11 = binary_repr(imm11, width11)
        self.assertEqual(binary_imm11, '0')
        self.assertEqual(width11, (upper-lower) + 1)

        upper = 30
        lower = 21
        imm10_1, width10_1 = bm.get_sub_bits_from_instr(vector, upper, lower)
        self.assertEqual(imm10_1, 956)
        binary_imm10_1 = binary_repr(imm10_1, width10_1)
        self.assertEqual(binary_imm10_1, '1110111100')
        self.assertEqual(width10_1, (upper-lower) + 1)

        concat_bits, width_jimm = bm.concat_bits([(imm20, width20), (imm19_12, width19_12), (imm11, width11), (imm10_1, width10_1)])
        binary_concat_bits = binary_repr(concat_bits, width_jimm)
        self.assertEqual(binary_concat_bits, '00011000001110111100')

        concat_sign_ext = bm.sign_extend_nbit_2_int32((concat_bits, width_jimm))
        self.assertEqual(concat_sign_ext, 99260)
        binary_conat_sign_ext = binary_repr(concat_sign_ext, width=32)
        self.assertEqual(binary_conat_sign_ext, '00000000000000011000001110111100')


class TestCPU(unittest.TestCase):

    def test_conditional_branch_equals_true(self):

        core = RV32ICORE() 
        bm = BitManip32()
        instr = bm.hex_str_2_unsigned_int("00208663")

        # Initialize core registers
        core.REG_FILE[1] = -5
        core.REG_FILE[2] = -5

        # PC offset
        offset = 6

        # Initialize memory with instruction
        core.memory[0] = instr

        # Single test run
        core.st_run()

        self.assertEqual(core.PC, offset)

    def test_conditional_branch_equals_false(self):

        core = RV32ICORE() 
        bm = BitManip32()
        instr = bm.hex_str_2_unsigned_int("00208663")

        # Initialize core registers
        core.REG_FILE[1] = -5
        core.REG_FILE[2] = -6

        # PC offset
        offset = 6

        # Initialize memory with instruction
        core.memory[0] = instr

        # Single test run
        core.st_run()

        self.assertEqual(core.PC, 4)

    def test_conditional_branch_not_equals_true(self):

        core = RV32ICORE() 
        bm = BitManip32()
        instr = bm.hex_str_2_unsigned_int("28301863")

        # Initialize core registers
        core.REG_FILE[0] = 0
        core.REG_FILE[3] = 6

        # PC offset
        offset = 328

        # Initialize memory with instruction
        core.memory[0] = instr

        # Single test run
        core.st_run()

        self.assertEqual(core.PC, offset)

    def test_conditional_branch_not_equals_false(self):

        core = RV32ICORE() 
        bm = BitManip32()
        instr = bm.hex_str_2_unsigned_int("28301863")

        # Initialize core registers
        core.REG_FILE[0] = 0
        core.REG_FILE[3] = 0

        # PC offset
        offset = 328

        # Initialize memory with instruction
        core.memory[0] = instr

        # Single test run
        core.st_run()

        self.assertEqual(core.PC, 4)

    def test_conditional_branch_lt_true(self):

        core = RV32ICORE() 
        bm = BitManip32()
        instr = bm.hex_str_2_unsigned_int("00054c63")

        # Initialize core registers
        core.REG_FILE[10] = -1
        core.REG_FILE[0] = 0

        # PC offset
        offset = 12

        # Initialize memory with instruction
        core.memory[0] = instr

        # Single test run
        core.st_run()

        self.assertEqual(core.PC, offset)

    def test_conditional_branch_lt_false(self):

        core = RV32ICORE() 
        bm = BitManip32()
        instr = bm.hex_str_2_unsigned_int("00054c63")

        # Initialize core registers
        core.REG_FILE[10] = 1
        core.REG_FILE[0] = 0

        # PC offset
        offset = 12

        # Initialize memory with instruction
        core.memory[0] = instr

        # Single test run
        core.st_run()

        self.assertEqual(core.PC, 4)

    def test_conditional_branch_gt_true(self):

        core = RV32ICORE() 
        bm = BitManip32()
        instr = bm.hex_str_2_unsigned_int("0020d663")

        # Initialize core registers
        core.REG_FILE[1] = 1
        core.REG_FILE[2] = -1

        # PC offset
        offset = 6

        # Initialize memory with instruction
        core.memory[0] = instr

        # Single test run
        core.st_run()

        self.assertEqual(core.PC, offset)

    def test_conditional_branch_gt_false(self):

        core = RV32ICORE() 
        bm = BitManip32()
        instr = bm.hex_str_2_unsigned_int("0020d663")

        # Initialize core registers
        core.REG_FILE[1] = -1
        core.REG_FILE[2] = 1

        # PC offset
        offset = 6

        # Initialize memory with instruction
        core.memory[0] = instr

        # Single test run
        core.st_run()

        self.assertEqual(core.PC, 4)

    def test_conditional_branch_ltu_true(self):

        core = RV32ICORE() 
        bm = BitManip32()
        instr = bm.hex_str_2_unsigned_int("0020e663")

        # Initialize core registers
        core.REG_FILE[1] = uint32(iinfo(uint32).max)-1
        core.REG_FILE[2] = uint32(iinfo(uint32).max)

        # PC offset
        offset = 6

        # Initialize memory with instruction
        core.memory[0] = instr

        # Single test run
        core.st_run()

        self.assertEqual(core.PC, offset)

    def test_conditional_branch_ltu_false(self):

        core = RV32ICORE() 
        bm = BitManip32()
        instr = bm.hex_str_2_unsigned_int("0020e663")

        # Initialize core registers
        core.REG_FILE[1] = uint32(iinfo(uint32).max)
        core.REG_FILE[2] = uint32(iinfo(uint32).max)

        # PC offset
        offset = 6

        # Initialize memory with instruction
        core.memory[0] = instr

        # Single test run
        core.st_run()

        self.assertEqual(core.PC, 4)

    def test_conditional_branch_geu_true(self):

        core = RV32ICORE() 
        bm = BitManip32()
        instr = bm.hex_str_2_unsigned_int("0020f663")

        # Initialize core registers
        core.REG_FILE[1] = uint32(iinfo(uint32).max)
        core.REG_FILE[2] = uint32(iinfo(uint32).max)-1

        # PC offset
        offset = 6

        # Initialize memory with instruction
        core.memory[0] = instr

        # Single test run
        core.st_run()

        self.assertEqual(core.PC, offset)

    def test_conditional_branch_geu_false(self):

        core = RV32ICORE() 
        bm = BitManip32()
        instr = bm.hex_str_2_unsigned_int("0020f663")

        # Initialize core registers
        core.REG_FILE[1] = uint32(iinfo(uint32).max)
        core.REG_FILE[2] = uint32(iinfo(uint32).max)

        # PC offset
        offset = 6

        # Initialize memory with instruction
        core.memory[0] = instr

        # Single test run
        core.st_run()

        self.assertEqual(core.PC, 4)

    def test_unknown_opcode1_0_value_error(self):

        core = RV32ICORE() 
        bm = BitManip32()
        instr = bm.hex_str_2_unsigned_int("0020f662")

        self.assertRaises(ValueError, core.decode, instr)

    def test_unknown_opcode6_2_value_error(self):

        core = RV32ICORE() 
        bm = BitManip32()
        instr = bm.hex_str_2_unsigned_int("0020f607")

        self.assertRaises(ValueError, core.decode, instr)

    def test_jump_and_link(self):

        core = RV32ICORE()
        bm = BitManip32()
        instr = bm.hex_str_2_unsigned_int("0100026f")

        PC_test = 0x4 * 20

        # Initialize PC
        core.PC = PC_test

        # PC offset
        offset = PC_test + 8

        # Initialize memory with instruction
        core.memory[core.PC] = instr

        # Single test run
        core.st_run()

        self.assertEqual(core.PC, offset)

        self.assertEqual(core.REG_FILE[4], PC_test + 0x4) 

    def test_jump_and_link_register(self):

        core = RV32ICORE()
        bm = BitManip32()
        instr = bm.hex_str_2_unsigned_int("000302e7")

        PC_test = 0x4 * 20

        # Initialize PC
        core.PC = PC_test

        core.REG_FILE[6] = uint32(13)

        # PC offset
        offset = 12

        # Initialize memory with instruction
        core.memory[core.PC] = instr

        # Single test run
        core.st_run()

        self.assertEqual(core.PC, offset)

        self.assertEqual(core.REG_FILE[5], PC_test + 0x4) 

    def test_load_upper_imm(self):

        core = RV32ICORE()
        bm = BitManip32()
        instr = bm.hex_str_2_unsigned_int("800002b7")

        PC_test = 0x4 * 20

        # Initialize PC
        core.PC = PC_test

        # upper immediate. pre-calculated
        u_imm = 2147483648

        # destination register
        dst = 5

        # Initialize memory with instruction
        core.memory[core.PC] = instr

        # Single test run
        core.st_run()

        self.assertEqual(core.PC, PC_test + 0x4)

        self.assertEqual(core.REG_FILE[dst], u_imm) 

    def test_add_upper_imm_2_pc(self):

        core = RV32ICORE()
        bm = BitManip32()
        instr = bm.hex_str_2_unsigned_int("00004517")

        PC_test = 0x4 * 20

        # Initialize PC
        core.PC = PC_test

        # upper immediate. pre-calculated
        u_imm =  16384 + PC_test

        # destination register
        dst = 10

        # Initialize memory with instruction
        core.memory[core.PC] = instr

        # Single test run
        core.st_run()

        self.assertEqual(core.PC, PC_test + 0x4)

        self.assertEqual(core.REG_FILE[dst], u_imm) 

    def test_reg_imm_add_positive(self):

        core = RV32ICORE()
        bm = BitManip32()
        instr = bm.hex_str_2_unsigned_int("00108713")

        PC_test = 0x4 * 20

        # Initialize PC
        core.PC = PC_test

        # Set up src register
        src_val = 5
        core.REG_FILE[1] = src_val

        # upper immediate. pre-calculated
        imm =  1

        # destination register
        dst = 14

        # Initialize memory with instruction
        core.memory[core.PC] = instr

        # Single test run
        core.st_run()

        self.assertEqual(core.PC, PC_test + 0x4)

        self.assertEqual(int32(core.REG_FILE[dst]), imm + src_val) 

    def test_reg_imm_add_negetive(self):

        core = RV32ICORE()
        bm = BitManip32()
        instr = bm.hex_str_2_unsigned_int("00108713")

        PC_test = 0x4 * 20

        # Initialize PC
        core.PC = PC_test

        # Set up src register
        src_val = -5
        core.REG_FILE[1] = src_val

        # upper immediate. pre-calculated
        imm =  1

        # destination register
        dst = 14

        # Initialize memory with instruction
        core.memory[core.PC] = instr

        # Single test run
        core.st_run()

        self.assertEqual(core.PC, PC_test + 0x4)

        self.assertEqual(int32(core.REG_FILE[dst]), imm + src_val) 

    def test_reg_imm_shift_left(self):

        core = RV32ICORE()
        bm = BitManip32()
        instr = bm.hex_str_2_unsigned_int("00709713")

        PC_test = 0x4 * 20

        # Initialize PC
        core.PC = PC_test

        # Set up src register
        src_val = uint32(iinfo(uint32).max)
        core.REG_FILE[1] = src_val

        # upper immediate. pre-calculated
        imm =  1

        # destination register
        dst = 14

        # Initialize memory with instruction
        core.memory[core.PC] = instr

        # Single test run
        core.st_run()

        self.assertEqual(core.PC, PC_test + 0x4)

        self.assertEqual(core.REG_FILE[dst], 0xffffff80) 

    def test_reg_imm_shift_right_logic(self):

        core = RV32ICORE()
        bm = BitManip32()
        instr = bm.hex_str_2_unsigned_int("0070d713")

        PC_test = 0x4 * 20

        # Initialize PC
        core.PC = PC_test

        # Set up src register
        src_val = uint32(iinfo(uint32).max)
        core.REG_FILE[1] = src_val

        # upper immediate. pre-calculated
        imm =  1

        # destination register
        dst = 14

        # Initialize memory with instruction
        core.memory[core.PC] = instr

        # Single test run
        core.st_run()

        self.assertEqual(core.PC, PC_test + 0x4)

        self.assertEqual(core.REG_FILE[dst], 0x01ffffff) 

    def test_reg_imm_shift_right_arith(self):
        core = RV32ICORE()
        bm = BitManip32()
        instr = bm.hex_str_2_unsigned_int("4070d713")

        PC_test = 0x4 * 20

        # Initialize PC
        core.PC = PC_test

        # Set up src register
        src_val = 0xf0ffffff
        core.REG_FILE[1] = src_val

        # upper immediate. pre-calculated
        imm =  1

        # destination register
        dst = 14

        # Initialize memory with instruction
        core.memory[core.PC] = instr

        # Single test run
        core.st_run()

        self.assertEqual(core.PC, PC_test + 0x4)

        self.assertEqual(core.REG_FILE[dst], 0xffe1ffff) 

    def test_shift_imm11_5_exception(self):

        core = RV32ICORE() 
        bm = BitManip32()
        exe_strat = RegImmInt_32()
        instr = bm.hex_str_2_unsigned_int("5070d713")

        self.assertRaises(ValueError, core.execute, instr, exe_strat)

    def test_reg_imm_slti_true(self):
        core = RV32ICORE()
        bm = BitManip32()
        instr = bm.hex_str_2_unsigned_int("0070a713")

        PC_test = 0x4 * 20

        # Initialize PC
        core.PC = PC_test

        # Set up src register
        src_val = -5
        core.REG_FILE[1] = src_val

        # upper immediate. pre-calculated
        imm =  1

        # destination register
        dst = 14

        # Initialize memory with instruction
        core.memory[core.PC] = instr

        # Single test run
        core.st_run()

        self.assertEqual(core.PC, PC_test + 0x4)

        self.assertEqual(core.REG_FILE[dst], 0x1) 

    def test_reg_imm_slti_false(self):
        core = RV32ICORE()
        bm = BitManip32()
        instr = bm.hex_str_2_unsigned_int("8000a713")

        PC_test = 0x4 * 20

        # Initialize PC
        core.PC = PC_test

        # Set up src register
        src_val = -5
        core.REG_FILE[1] = src_val

        # upper immediate. pre-calculated
        imm =  1

        # destination register
        dst = 14

        # Initialize memory with instruction
        core.memory[core.PC] = instr

        # Single test run
        core.st_run()

        self.assertEqual(core.PC, PC_test + 0x4)

        self.assertEqual(core.REG_FILE[dst], 0x0) 

    def test_reg_imm_sltiu_true(self):
        core = RV32ICORE()
        bm = BitManip32()
        instr = bm.hex_str_2_unsigned_int("8000b713")

        PC_test = 0x4 * 20

        # Initialize PC
        core.PC = PC_test

        # Set up src register
        src_val = 7
        core.REG_FILE[1] = src_val

        # upper immediate. pre-calculated
        imm =  1

        # destination register
        dst = 14

        # Initialize memory with instruction
        core.memory[core.PC] = instr

        # Single test run
        core.st_run()

        self.assertEqual(core.PC, PC_test + 0x4)

        self.assertEqual(core.REG_FILE[dst], 0x1) 

    def test_reg_imm_sltiu_false(self):
        core = RV32ICORE()
        bm = BitManip32()
        instr = bm.hex_str_2_unsigned_int("0070b713")

        PC_test = 0x4 * 20

        # Initialize PC
        core.PC = PC_test

        # Set up src register
        src_val = -5
        core.REG_FILE[1] = src_val

        # upper immediate. pre-calculated
        imm =  1

        # destination register
        dst = 14

        # Initialize memory with instruction
        core.memory[core.PC] = instr

        # Single test run
        core.st_run()

        self.assertEqual(core.PC, PC_test + 0x4)

        self.assertEqual(core.REG_FILE[dst], 0x0) 

    def test_reg_imm_xor(self):
        core = RV32ICORE()
        bm = BitManip32()
        instr = bm.hex_str_2_unsigned_int("f0f0c713")

        PC_test = 0x4 * 20

        # Initialize PC
        core.PC = PC_test

        # Set up src register
        src_val = -5
        core.REG_FILE[1] = src_val

        # upper immediate. pre-calculated
        imm =  1

        # destination register
        dst = 14

        # Initialize memory with instruction
        core.memory[core.PC] = instr

        # Single test run
        core.st_run()

        self.assertEqual(core.PC, PC_test + 0x4)

        self.assertEqual(core.REG_FILE[dst], 0xf4) 

    def test_reg_imm_or(self):
        core = RV32ICORE()
        bm = BitManip32()
        instr = bm.hex_str_2_unsigned_int("f0f0e713")

        PC_test = 0x4 * 20

        # Initialize PC
        core.PC = PC_test

        # Set up src register
        src_val = -5
        core.REG_FILE[1] = src_val

        # upper immediate. pre-calculated
        imm =  1

        # destination register
        dst = 14

        # Initialize memory with instruction
        core.memory[core.PC] = instr

        # Single test run
        core.st_run()

        self.assertEqual(core.PC, PC_test + 0x4)

        self.assertEqual(core.REG_FILE[dst], 0xffffffff) 

    def test_reg_imm_and(self):
        core = RV32ICORE()
        bm = BitManip32()
        instr = bm.hex_str_2_unsigned_int("f0f0f713")

        PC_test = 0x4 * 20

        # Initialize PC
        core.PC = PC_test

        # Set up src register
        src_val = -5
        core.REG_FILE[1] = src_val

        # upper immediate. pre-calculated
        imm =  1

        # destination register
        dst = 14

        # Initialize memory with instruction
        core.memory[core.PC] = instr

        # Single test run
        core.st_run()

        self.assertEqual(core.PC, PC_test + 0x4)

        self.assertEqual(core.REG_FILE[dst], 0xffffff0b) 

    def test_reg_reg_add_positive(self):

        core = RV32ICORE()
        bm = BitManip32()
        instr = bm.hex_str_2_unsigned_int("00208733")

        PC_test = 0x4 * 20

        # Initialize PC
        core.PC = PC_test

        # Set up src register
        src1_val = 5
        core.REG_FILE[1] = src1_val

        src2_val = 4
        core.REG_FILE[2] = src2_val

        # destination register
        dst = 14

        # Initialize memory with instruction
        core.memory[core.PC] = instr

        # Single test run
        core.st_run()

        self.assertEqual(core.PC, PC_test + 0x4)

        self.assertEqual(int32(core.REG_FILE[dst]), src1_val + src2_val) 

    def test_reg_reg_add_negetive(self):

        core = RV32ICORE()
        bm = BitManip32()
        instr = bm.hex_str_2_unsigned_int("00208733")

        PC_test = 0x4 * 20

        # Initialize PC
        core.PC = PC_test

        # Set up src register
        src1_val = -5
        core.REG_FILE[1] = src1_val

        src2_val = -4
        core.REG_FILE[2] = src2_val

        # destination register
        dst = 14

        # Initialize memory with instruction
        core.memory[core.PC] = instr

        # Single test run
        core.st_run()

        self.assertEqual(core.PC, PC_test + 0x4)

        self.assertEqual(int32(core.REG_FILE[dst]), -9) 

    def test_reg_reg_sub_positive(self):

        core = RV32ICORE()
        bm = BitManip32()
        instr = bm.hex_str_2_unsigned_int("40208733")

        PC_test = 0x4 * 20

        # Initialize PC
        core.PC = PC_test

        # Set up src register
        src1_val = 5
        core.REG_FILE[1] = src1_val

        src2_val = 4
        core.REG_FILE[2] = src2_val

        # destination register
        dst = 14

        # Initialize memory with instruction
        core.memory[core.PC] = instr

        # Single test run
        core.st_run()

        self.assertEqual(core.PC, PC_test + 0x4)

        self.assertEqual(int32(core.REG_FILE[dst]), src1_val - src2_val) 

    def test_reg_reg_add_negetive(self):

        core = RV32ICORE()
        bm = BitManip32()
        instr = bm.hex_str_2_unsigned_int("40208733")

        PC_test = 0x4 * 20

        # Initialize PC
        core.PC = PC_test

        # Set up src register
        src1_val = -5
        core.REG_FILE[1] = src1_val

        src2_val = -4
        core.REG_FILE[2] = src2_val

        # destination register
        dst = 14

        # Initialize memory with instruction
        core.memory[core.PC] = instr

        # Single test run
        core.st_run()

        self.assertEqual(core.PC, PC_test + 0x4)

        self.assertEqual(int32(core.REG_FILE[dst]), -1) 

    def test_shift_funct7_exception_reg_reg_int(self):

        core = RV32ICORE() 
        bm = BitManip32()
        exe_strat = RegRegInt_32()
        instr = bm.hex_str_2_unsigned_int("50208733")

        self.assertRaises(ValueError, core.execute, instr, exe_strat)

    def test_reg_reg_shift_left(self):

        core = RV32ICORE()
        bm = BitManip32()
        instr = bm.hex_str_2_unsigned_int("00209733")

        PC_test = 0x4 * 20

        # Initialize PC
        core.PC = PC_test

        # Set up src register
        src1_val = 1
        core.REG_FILE[1] = src1_val

        src2_val = 4
        core.REG_FILE[2] = src2_val

        # destination register
        dst = 14

        # Initialize memory with instruction
        core.memory[core.PC] = instr

        # Single test run
        core.st_run()

        self.assertEqual(core.PC, PC_test + 0x4)

        self.assertEqual(core.REG_FILE[dst], 0x10) 

    def test_reg_reg_shift_right_logic(self):

        core = RV32ICORE()
        bm = BitManip32()
        instr = bm.hex_str_2_unsigned_int("0020d733")

        PC_test = 0x4 * 20

        # Initialize PC
        core.PC = PC_test

        # Set up src register
        src1_val = uint32(iinfo(uint32).max)
        core.REG_FILE[1] = src1_val

        src2_val = 4
        core.REG_FILE[2] = src2_val

        # destination register
        dst = 14

        # Initialize memory with instruction
        core.memory[core.PC] = instr

        # Single test run
        core.st_run()

        self.assertEqual(core.PC, PC_test + 0x4)

        self.assertEqual(core.REG_FILE[dst], 0x0fffffff) 

    def test_reg_reg_shift_right_arith(self):
        core = RV32ICORE()
        bm = BitManip32()
        instr = bm.hex_str_2_unsigned_int("4020d733")

        PC_test = 0x4 * 20

        # Initialize PC
        core.PC = PC_test

        # Set up src register
        src1_val = 0xf0ffffff
        core.REG_FILE[1] = src1_val

        src2_val = 7
        core.REG_FILE[2] = src2_val
        # Set up src register

        # destination register
        dst = 14

        # Initialize memory with instruction
        core.memory[core.PC] = instr

        # Single test run
        core.st_run()

        self.assertEqual(core.PC, PC_test + 0x4)

        self.assertEqual(core.REG_FILE[dst], 0xffe1ffff) 

    def test_shift_imm11_5_exception_reg_reg_int(self):

        core = RV32ICORE() 
        bm = BitManip32()
        exe_strat = RegRegInt_32()
        instr = bm.hex_str_2_unsigned_int("5070d733")

        self.assertRaises(ValueError, core.execute, instr, exe_strat)

    def test_reg_reg_slti_true(self):
        core = RV32ICORE()
        bm = BitManip32()
        instr = bm.hex_str_2_unsigned_int("0020a733")

        PC_test = 0x4 * 20

        # Initialize PC
        core.PC = PC_test

        # Set up src register
        src1_val = -5
        core.REG_FILE[1] = src1_val

        src2_val = 7
        core.REG_FILE[2] = src2_val

        # destination register
        dst = 14

        # Initialize memory with instruction
        core.memory[core.PC] = instr

        # Single test run
        core.st_run()

        self.assertEqual(core.PC, PC_test + 0x4)

        self.assertEqual(core.REG_FILE[dst], 0x1) 

    def test_reg_reg_slti_false(self):
        core = RV32ICORE()
        bm = BitManip32()
        instr = bm.hex_str_2_unsigned_int("0020a733")

        PC_test = 0x4 * 20

        # Initialize PC
        core.PC = PC_test

        # Set up src register
        src1_val = 7
        core.REG_FILE[1] = src1_val

        src2_val = -5
        core.REG_FILE[2] = src2_val

        # destination register
        dst = 14

        # Initialize memory with instruction
        core.memory[core.PC] = instr

        # Single test run
        core.st_run()

        self.assertEqual(core.PC, PC_test + 0x4)

        self.assertEqual(core.REG_FILE[dst], 0x0) 

    def test_reg_reg_sltiu_true(self):
        core = RV32ICORE()
        bm = BitManip32()
        instr = bm.hex_str_2_unsigned_int("0020b733")

        PC_test = 0x4 * 20

        # Initialize PC
        core.PC = PC_test

        # Set up src register
        src1_val = 7
        core.REG_FILE[1] = src1_val

        src2_val = -5
        core.REG_FILE[2] = src2_val

        # destination register
        dst = 14

        # Initialize memory with instruction
        core.memory[core.PC] = instr

        # Single test run
        core.st_run()

        self.assertEqual(core.PC, PC_test + 0x4)

        self.assertEqual(core.REG_FILE[dst], 0x1) 

    def test_reg_reg_sltiu_false(self):
        core = RV32ICORE()
        bm = BitManip32()
        instr = bm.hex_str_2_unsigned_int("0070b733")

        PC_test = 0x4 * 20

        # Initialize PC
        core.PC = PC_test

        # Set up src register
        src1_val = -5
        core.REG_FILE[1] = src1_val

        src2_val = 7
        core.REG_FILE[2] = src2_val

        # destination register
        dst = 14

        # Initialize memory with instruction
        core.memory[core.PC] = instr

        # Single test run
        core.st_run()

        self.assertEqual(core.PC, PC_test + 0x4)

        self.assertEqual(core.REG_FILE[dst], 0x0) 

    def test_reg_reg_xor(self):
        core = RV32ICORE()
        bm = BitManip32()
        instr = bm.hex_str_2_unsigned_int("0020c733")

        PC_test = 0x4 * 20

        # Initialize PC
        core.PC = PC_test

        # Set up src register
        src1_val = -5
        core.REG_FILE[1] = src1_val

        src2_val = -241
        core.REG_FILE[2] = src2_val

        # destination register
        dst = 14

        # Initialize memory with instruction
        core.memory[core.PC] = instr

        # Single test run
        core.st_run()

        self.assertEqual(core.PC, PC_test + 0x4)

        self.assertEqual(core.REG_FILE[dst], 0xf4) 

    def test_reg_reg_or(self):
        core = RV32ICORE()
        bm = BitManip32()
        instr = bm.hex_str_2_unsigned_int("0020e733")

        PC_test = 0x4 * 20

        # Initialize PC
        core.PC = PC_test

        # Set up src register
        src1_val = -5
        core.REG_FILE[1] = src1_val

        src2_val = -241
        core.REG_FILE[2] = src2_val

        # destination register
        dst = 14

        # Initialize memory with instruction
        core.memory[core.PC] = instr

        # Single test run
        core.st_run()

        self.assertEqual(core.PC, PC_test + 0x4)

        self.assertEqual(core.REG_FILE[dst], 0xffffffff) 

    def test_reg_reg_and(self):
        core = RV32ICORE()
        bm = BitManip32()
        instr = bm.hex_str_2_unsigned_int("0020f733")

        PC_test = 0x4 * 20

        # Initialize PC
        core.PC = PC_test

        # Set up src register
        src1_val = -5
        core.REG_FILE[1] = src1_val

        src2_val = -241
        core.REG_FILE[2] = src2_val

        # destination register
        dst = 14

        # Initialize memory with instruction
        core.memory[core.PC] = instr

        # Single test run
        core.st_run()

        self.assertEqual(core.PC, PC_test + 0x4)

        self.assertEqual(core.REG_FILE[dst], 0xffffff0b) 


unittest.main()