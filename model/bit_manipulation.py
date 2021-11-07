from numpy import int32, uint, uint32, iinfo
from numpy import bitwise_or
from numpy import left_shift, right_shift
from numpy import binary_repr
from typing import List

class BitManip32:
	""" Basic HW developer friendly bit manipulation class """

	ui32_max = uint32(iinfo(uint32).max)
	ii32_max = int32(iinfo(int32).max)
	ii32_min = int32(iinfo(int32).min)

	# Concats bit vectors. Provide list of tuples as input. where: a is uint32 value, width = len(bin(a))
	# args = [(input, width), (input, width), .....]
	def concat_bits(self, args : List[tuple]) -> tuple:
		concat_str = ""
		for x in args:
			concat_str += binary_repr(x[0], width=x[1])

		return uint32(int(concat_str, 2)), len(concat_str)
		
	# Gets a bit vector from upper to lower. Returns a int32 value. Input index should match verilog spec
	# For example : input[7:0] -> 8 bit vector. provide upper = 7, lower = 0. 
	def get_sub_bits_from_instr(self, instr: uint32, upper : int, lower : int = 0) -> tuple:

		if upper < lower:
			raise ValueError("Upper cannot be less than lower")
		
		# discart MSB bits till "upper"
		instr = uint32(left_shift(instr, (31 - upper)))

		# discard LSB bits upto "lower"
		bit_vector = right_shift(instr, ((31-upper) + lower))

		return (bit_vector, (upper-lower) + 1)

	def sign_extend_nbit_2_int32(self, args : tuple) -> int32:
		vector = args[0]
		width = args[1]

		last_bit = uint32(right_shift(vector, width-1))

		mask = uint32(left_shift((self.ui32_max * last_bit), width))

		return int32(bitwise_or(mask, vector))

	def sign_extend_nbit_2_uint32(self, args : tuple) -> uint32:
		vector = args[0]
		width = args[1]

		last_bit = uint32(right_shift(vector, width-1))

		mask = uint32(left_shift((self.ui32_max * last_bit), width))

		return uint32(bitwise_or(mask, vector))

	def hex_str_2_unsigned_int(self, hex_str : str) -> uint32:
		if len(hex_str) != 8:
			raise ValueError(f"hex_str = {hex_str} is not 32 bits long")
		
		# Hex string to unsigned 32 bit integer
		return uint32(int(hex_str, 16))