from numpy import uint64, int64, int32, uint, uint32, iinfo
from numpy import bitwise_or
from numpy import left_shift, right_shift
from numpy import binary_repr
from typing import List
from enum import Enum

class XLen(Enum):
	_32BIT = 32
	_64BIT = 64


class BitManip:
	""" Basic HW developer friendly bit manipulation class """

	ui32_max: uint32 = uint32(iinfo(uint32).max)
	ui64_max: uint64 = uint64(iinfo(uint64).max)
	ii32_max: int32 = int32(iinfo(int32).max)
	ii32_min: int32 = int32(iinfo(int32).min)
	ii64_max: int64 = int64(iinfo(int64).max)
	ii64_min: int64 = int64(iinfo(int64).min)

	def __init__(self, xlen=XLen._32BIT):
		self.xlen: Enum = xlen.value

		if self.xlen == XLen._64BIT.value:
			self.utype = uint64
			self.stype = int64
			self.utype_max = self.ui64_max
			self.stype_max = self.ii64_max
			self.stype_min = self.ii64_min

		self.utype = uint32
		self.utype_max = self.ui32_max
		self.stype_max = self.ii32_max
		self.stype_min = self.ii32_min
		self.stype = int32

	def concat_bits(self, args : List[tuple]) -> tuple:
		""" Concats bit vectors. Provide list of tuples as input. 
		where a is uint32 value, width = len(bin(a))
		"""
		concat_str = ""
		for x in args:
			concat_str += binary_repr(x[0], width=x[1])

		return self.utype(int(concat_str, 2)), len(concat_str)
		
	def get_sub_bits_from_instr(
		self, instr: uint32, upper : int, lower : int = 0) -> tuple:
		""" Gets a bit vector from upper to lower. Returns a signed value. 
		Input index should match verilog spec.
		For example : input[7:0] -> 8 bit vector. provide upper = 7, lower = 0. 
		"""

		if upper < lower:
			raise ValueError("Upper cannot be less than lower")
		
		# discart MSB bits till "upper"
		instr = self.utype(left_shift(instr, ((self.xlen-1)- upper)))

		# discard LSB bits upto "lower"
		bit_vector = right_shift(instr, (((self.xlen-1)-upper) + lower))

		return (bit_vector, (upper-lower) + 1)

	def sign_extend_nbit_2_int(self, args : tuple):
		vector = args[0]
		width = args[1]

		last_bit = self.utype(right_shift(vector, width-1))

		mask = self.utype(left_shift((self.utype_max * last_bit), width))

		return self.stype(bitwise_or(mask, vector))

	def sign_extend_nbit_2_unsigned_int(self, args : tuple):
		vector = args[0]
		width = args[1]

		last_bit = self.utype(right_shift(vector, width-1))

		mask = self.utype(left_shift((self.utype_max * last_bit), width))

		return self.utype(bitwise_or(mask, vector))

	def hex_str_2_unsigned_int(self, hex_str : str):
		if len(hex_str) != 8:
			raise ValueError(f"hex_str = {hex_str} is not 32 bits long")
		
		# Hex string to unsigned 32 bit integer
		return self.utype(int(hex_str, 16))