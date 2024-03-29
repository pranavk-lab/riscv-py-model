#!/usr/bin/env python3
from logging import setLogRecordFactory
from context import BitManip, XLen
from bit_manipulation import BitManip, XLen
from numpy import binary_repr, isfinite, issubdtype, uint32, int32, iinfo
from numpy import base_repr
import unittest
import coverage


class TestBitManip32(unittest.TestCase):

    hex_str_high_msb = "f7830000"
    hex_str_low_msb = "77830521"

    def test_hex_str_2_unsigned_int(self):
        bm = BitManip(XLen._32BIT)
        hex_str = self.hex_str_high_msb
        vector = bm.hex_str_2_unsigned_int(hex_str)
        self.assertEqual(issubdtype(vector, uint32), True)
        
    def test_get_sub_bits_from_instr(self):
        # Test get_sub_bits_from_instr 
        upper = 31
        lower = 12
        bm = BitManip(XLen._32BIT)
        hex_str = self.hex_str_high_msb
        vector = bm.hex_str_2_unsigned_int(hex_str)
        vector_unsigned, width = (
            bm.get_sub_bits_from_instr(vector, upper, lower)
        )
        self.assertEqual(vector_unsigned, 1013808)
        vector_unsigned_binary = binary_repr(vector_unsigned, width=width)
        self.assertEqual(vector_unsigned_binary, '11110111100000110000')
        self.assertRaises(
            ValueError, bm.hex_str_2_unsigned_int, 'f' + self.hex_str_low_msb
        )
        self.assertEqual(width, (upper-lower) + 1)

    def test_sub_bits_from_instr_negetive_number(self):
        # Test get_sub_bits_from_instr negetive number
        upper = 31
        lower = 12
        bm = BitManip(XLen._32BIT)
        hex_str = self.hex_str_high_msb
        vector = bm.hex_str_2_unsigned_int(hex_str)
        vector_signed = bm.sign_extend_nbit_2_int(
            bm.get_sub_bits_from_instr(vector, upper, lower)
        )
        self.assertEqual(vector_signed, -34768)
        vector_signed_binary = binary_repr(vector_signed, width=32)
        self.assertEqual(
            vector_signed_binary, '11111111111111110111100000110000'
        )
        self.assertRaises(
            ValueError, bm.get_sub_bits_from_instr, vector, lower, upper
        )

    def test_get_sub_bits_from_instr_positive_number(self):
        # Test get_sub_bits_from_instr positive number
        upper = 31
        lower = 12
        bm = BitManip(XLen._32BIT)
        hex_str = self.hex_str_high_msb
        vector = bm.hex_str_2_unsigned_int(hex_str)
        vector_signed = bm.sign_extend_nbit_2_unsigned_int(
            bm.get_sub_bits_from_instr(vector, upper, lower)
        )
        self.assertEqual(vector_signed, 4294932528)
        vector_signed_binary = binary_repr(vector_signed, width=32) 
        self.assertEqual(
            vector_signed_binary, '11111111111111110111100000110000'
        )

    def test_concat_S_type_imm(self):
        upper = 31
        lower = 25
        bm = BitManip(XLen._32BIT)
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

        unsigned_concat_bits, width_simm = (
            bm.concat_bits([(imm11_5, width11_5), (imm4_0, width4_0)])
        )
        concat_bits = bm.sign_extend_nbit_2_int(
            (unsigned_concat_bits, width_simm)
        )
        binary_concat_bits = binary_repr(concat_bits, width_simm)
        self.assertEqual(concat_bits, 1898)
        self.assertEqual(binary_concat_bits, '011101101010')

    def test_concat_J_imm(self):
        upper = 31
        lower = 31
        bm = BitManip(XLen._32BIT)
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

        concat_bits, width_jimm = bm.concat_bits(
            [(imm20, width20), (imm19_12, width19_12), 
            (imm11, width11), (imm10_1, width10_1)]
        )
        binary_concat_bits = binary_repr(concat_bits, width_jimm)
        self.assertEqual(binary_concat_bits, '00011000001110111100')

        concat_sign_ext = bm.sign_extend_nbit_2_int((concat_bits, width_jimm))
        self.assertEqual(concat_sign_ext, 99260)
        binary_conat_sign_ext = binary_repr(concat_sign_ext, width=32)
        self.assertEqual(
            binary_conat_sign_ext, '00000000000000011000001110111100'
        )


class TestBitManip64(unittest.TestCase):

    hex_str_high_msb = "f7830000"
    hex_str_low_msb = "77830521"

    def test_hex_str_2_unsigned_int(self):
        bm = BitManip(XLen._64BIT)
        hex_str = self.hex_str_high_msb
        vector = bm.hex_str_2_unsigned_int(hex_str)
        self.assertEqual(issubdtype(vector, bm.uint), True)
        
    def test_get_sub_bits_from_instr(self):
        # Test get_sub_bits_from_instr 
        upper = 31
        lower = 12
        bm = BitManip(XLen._64BIT)
        hex_str = self.hex_str_high_msb
        vector = bm.hex_str_2_unsigned_int(hex_str)
        vector_unsigned, width = (
            bm.get_sub_bits_from_instr(vector, upper, lower)
        )
        self.assertEqual(vector_unsigned, 1013808)
        vector_unsigned_binary = binary_repr(vector_unsigned, width=width)
        self.assertEqual(vector_unsigned_binary, '11110111100000110000')
        self.assertRaises(
            ValueError, bm.hex_str_2_unsigned_int, 
            'f' + base_repr(bm.uint_max, bm.uint(16))
        )
        self.assertEqual(width, (upper-lower) + 1)

    def test_sub_bits_from_instr_negetive_number(self):
        # Test get_sub_bits_from_instr negetive number
        upper = 31
        lower = 12
        bm = BitManip(XLen._64BIT)
        hex_str = self.hex_str_high_msb
        vector = bm.hex_str_2_unsigned_int(hex_str)
        vector_signed = bm.sign_extend_nbit_2_int(
            bm.get_sub_bits_from_instr(vector, upper, lower)
        )
        self.assertEqual(vector_signed, -34768)
        vector_signed_binary = binary_repr(vector_signed, width=32)
        self.assertEqual(
            vector_signed_binary, '11111111111111110111100000110000'
        )
        self.assertRaises(
            ValueError, bm.get_sub_bits_from_instr, vector, lower, upper
        )

    def test_get_sub_bits_from_instr_positive_number(self):
        # Test get_sub_bits_from_instr positive number
        upper = 31
        lower = 12
        bm = BitManip(XLen._64BIT)
        hex_str = self.hex_str_high_msb
        vector = bm.hex_str_2_unsigned_int(hex_str)
        vector_signed = bm.sign_extend_nbit_2_unsigned_int(
            bm.get_sub_bits_from_instr(vector, upper, lower)
        )
        self.assertEqual(vector_signed, 18446744073709516848)
        vector_signed_binary = binary_repr(vector_signed, bm.xlen) 
        self.assertEqual(
            vector_signed_binary, 
            '1111111111111111111111111111111111111111111111110111100000110000'
        )

    def test_concat_S_type_imm(self):
        upper = 31
        lower = 25
        bm = BitManip(XLen._64BIT)
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

        unsigned_concat_bits, width_simm = (
            bm.concat_bits([(imm11_5, width11_5), (imm4_0, width4_0)])
        )
        concat_bits = bm.sign_extend_nbit_2_int(
            (unsigned_concat_bits, width_simm)
        )
        binary_concat_bits = binary_repr(concat_bits, width_simm)
        self.assertEqual(concat_bits, 1898)
        self.assertEqual(binary_concat_bits, '011101101010')

    def test_concat_J_imm(self):
        upper = 31
        lower = 31
        bm = BitManip(XLen._64BIT)
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

        concat_bits, width_jimm = bm.concat_bits(
            [(imm20, width20), (imm19_12, width19_12), 
            (imm11, width11), (imm10_1, width10_1)]
        )
        binary_concat_bits = binary_repr(concat_bits, width_jimm)
        self.assertEqual(binary_concat_bits, '00011000001110111100')

        concat_sign_ext = bm.sign_extend_nbit_2_int((concat_bits, width_jimm))
        self.assertEqual(concat_sign_ext, 99260)
        binary_conat_sign_ext = binary_repr(concat_sign_ext, width=32)
        self.assertEqual(
            binary_conat_sign_ext, '00000000000000011000001110111100'
        )

if __name__ == "__main__":
    unittest.main()