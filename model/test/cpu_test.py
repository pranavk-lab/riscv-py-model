#!/usr/bin/env python3
import sys
sys.path.append('../')
from bit_manipulation import BitManip_32 
from numpy import binary_repr, issubdtype, uint32, int32
import unittest
import coverage


class TestBitManip_32(unittest.TestCase):

    hex_str_high_msb = "f7830000"
    hex_str_low_msb = "77830521"

    def test_hex_str_2_unsigned_int(self):
        bm = BitManip_32()
        hex_str = self.hex_str_high_msb
        vector = bm.hex_str_2_unsigned_int(hex_str)
        self.assertEqual(issubdtype(vector, uint32), True)
        
    def test_get_sub_bits_from_instr(self):
        # Test get_sub_bits_from_instr 
        upper = 31
        lower = 12
        bm = BitManip_32()
        hex_str = self.hex_str_high_msb
        vector = bm.hex_str_2_unsigned_int(hex_str)
        vector_unsigned = bm.get_sub_bits_from_instr(vector, upper, lower)
        self.assertEqual(vector_unsigned, 1013808)
        vector_unsigned_binary = binary_repr(vector_unsigned, width=((upper-lower) + 1))
        self.assertEqual(vector_unsigned_binary, '11110111100000110000')
        self.assertRaises(ValueError, bm.hex_str_2_unsigned_int, 'f' + self.hex_str_low_msb)

    def test_sub_bits_from_instr_negetive_number(self):
        # Test get_sub_bits_from_instr negetive number
        upper = 31
        lower = 12
        bm = BitManip_32()
        hex_str = self.hex_str_high_msb
        vector = bm.hex_str_2_unsigned_int(hex_str)
        vector_signed = bm.sign_extend_nbit_2_32bit(bm.get_sub_bits_from_instr(vector, upper, lower), ((upper-lower) + 1))
        self.assertEqual(vector_signed, -34768)
        vector_signed_binary = binary_repr(vector_signed, width=32)
        self.assertEqual(vector_signed_binary, '11111111111111110111100000110000')
        self.assertRaises(ValueError, bm.get_sub_bits_from_instr, vector, lower, upper)

    def test_get_sub_bits_from_instr_positive_number(self):
        # Test get_sub_bits_from_instr positive number
        upper = 31
        lower = 12
        bm = BitManip_32()
        hex_str = self.hex_str_low_msb
        vector = bm.hex_str_2_unsigned_int(hex_str)
        vector_signed = bm.sign_extend_nbit_2_32bit(bm.get_sub_bits_from_instr(vector, upper, lower), ((upper-lower) + 1))
        self.assertEqual(vector_signed, 489520)
        vector_signed_binary = binary_repr(vector_signed, width=32) 
        self.assertEqual(vector_signed_binary, '00000000000001110111100000110000')

    def test_concat_S_type_imm(self):
        upper = 31
        lower = 25
        bm = BitManip_32()
        hex_str = self.hex_str_low_msb
        vector = bm.hex_str_2_unsigned_int(hex_str)
        imm11_5 = bm.get_sub_bits_from_instr(vector, upper, lower)
        self.assertEqual(imm11_5, 59)
        binary_imm11_5 = binary_repr(imm11_5, width=((upper-lower) + 1))
        self.assertEqual(binary_imm11_5, '0111011')

        upper = 11
        lower = 7
        vector = bm.hex_str_2_unsigned_int(hex_str)
        imm4_0 = bm.get_sub_bits_from_instr(vector, upper, lower)
        self.assertEqual(imm4_0, 10)
        binary_imm4_0 = binary_repr(imm4_0, width=((upper-lower) + 1))
        self.assertEqual(binary_imm4_0, '01010')

        concat_bits = bm.sign_extend_nbit_2_32bit(bm.concat_bits([(imm11_5, 31-25 + 1), (imm4_0, 11-7 + 1)]), 12)
        binary_concat_bits = binary_repr(concat_bits, width=12)
        self.assertEqual(concat_bits, 1898)
        self.assertEqual(binary_concat_bits, '011101101010')

    def test_concat_J_imm(self):
        upper = 31
        lower = 31
        bm = BitManip_32()
        hex_str = self.hex_str_low_msb
        vector = bm.hex_str_2_unsigned_int(hex_str)
        imm20 = bm.get_sub_bits_from_instr(vector, upper, lower)
        self.assertEqual(imm20, 0)
        binary_imm20 = binary_repr(imm20, width=((upper-lower) + 1))
        self.assertEqual(binary_imm20, '0')

        upper = 19
        lower = 12 
        imm19_12 = bm.get_sub_bits_from_instr(vector, upper, lower)
        self.assertEqual(imm19_12, 48)
        binary_imm19_12 = binary_repr(imm19_12, width=((upper-lower) + 1))
        self.assertEqual(binary_imm19_12, '00110000')

        upper = 20
        lower = 20
        imm11 = bm.get_sub_bits_from_instr(vector, upper, lower)
        self.assertEqual(imm11, 0)
        binary_imm11 = binary_repr(imm11, width=((upper-lower) + 1))
        self.assertEqual(binary_imm11, '0')

        upper = 30
        lower = 21
        imm10_1 = bm.get_sub_bits_from_instr(vector, upper, lower)
        self.assertEqual(imm10_1, 956)
        binary_imm10_1 = binary_repr(imm10_1, width=((upper-lower) + 1))
        self.assertEqual(binary_imm10_1, '1110111100')

        concat_bits = bm.concat_bits([(imm20, (31-31) + 1), (imm19_12, (19-12) + 1), (imm11, (20-20) +1), (imm10_1, (30-21) +1)])
        binary_concat_bits = binary_repr(concat_bits, width=20)
        self.assertEqual(binary_concat_bits, '00011000001110111100')

        concat_sign_ext = bm.sign_extend_nbit_2_32bit(concat_bits, 20)
        self.assertEqual(concat_sign_ext, 99260)
        binary_conat_sign_ext = binary_repr(concat_sign_ext, width=32)
        self.assertEqual(binary_conat_sign_ext, '00000000000000011000001110111100')

unittest.main()