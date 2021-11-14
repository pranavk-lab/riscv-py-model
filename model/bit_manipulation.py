#!/usr/bin/env python3
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
		self.xlen: int= xlen.value

		self.uint = uint32
		self.uint_max = self.ui32_max
		self.int_max = self.ii32_max
		self.int_min = self.ii32_min
		self.int = int32

		if self.xlen == XLen._64BIT.value:
			self.uint = uint64
			self.int = int64
			self.uint_max = self.ui64_max
			self.int_max = self.ii64_max
			self.int_min = self.ii64_min


	def concat_bits(self, args : List[tuple]) -> tuple:
		""" Concats bit vectors. Provide list of tuples as input. 
		where a is uint32 value, width = len(bin(a))
		"""
		concat_str = ""
		for x in args:
			concat_str += binary_repr(x[0], width=x[1])

		return self.uint(int(concat_str, 2)), len(concat_str)
		
	def get_sub_bits_from_instr(
		self, instr: uint, upper : int, lower : int = 0) -> tuple:
		""" Gets a bit vector from upper to lower. Returns a signed value. 
		Input index should match verilog spec.
		For example : input[7:0] -> 8 bit vector. provide upper = 7, lower = 0. 
		"""

		if upper < lower:
			raise ValueError("Upper cannot be less than lower")

		if upper >= self.xlen:
			raise ValueError("upper cannot be out of bounds")

		# discart MSB bits till "upper"
		instr = self.uint(
			left_shift(self.uint(instr), self.uint(((self.xlen-1)- upper)))
		)

		# discard LSB bits upto "lower"
		bit_vector = right_shift(
			self.uint(instr), self.uint((((self.xlen-1)-upper) + lower))
		)

		return (bit_vector, (upper-lower) + 1)

	def sign_extend_nbit_2_int(self, args : tuple):
		return self.int(self.sign_extend_nbit_2_unsigned_int(args))

	def sign_extend_nbit_2_unsigned_int(self, args : tuple):
		vector = args[0]
		width = args[1]

		last_bit = self.uint(right_shift(self.uint(vector), self.uint(width-1)))

		mask = self.uint(
			left_shift((self.uint_max * last_bit), self.uint(width))
		)

		return self.uint(bitwise_or(mask, vector))

	def hex_str_2_unsigned_int(self, hex_str : str):
		if len(hex_str) > self.xlen//4:
			raise ValueError(
				f"hex_str = {hex_str} is greater than {self.xlen} bits long"
			)
		
		# Hex string to unsigned xlen bit integer
		return self.uint(int(hex_str, 16))


def main():
	uint64_vector = "fffffffffffffff"

	upper = 31
	lower = 31
	bm = BitManip(XLen._64BIT)
	hex_str = uint64_vector
	vector = bm.hex_str_2_unsigned_int(hex_str)
	imm20, width20 = bm.get_sub_bits_from_instr(vector, upper, lower)
	# bm = BitManip(XLen._64BIT)
	# vector = bm.hex_str_2_unsigned_int(uint64_vector)

	print(binary_repr(vector, 64))
	print(imm20)


if __name__ == "__main__":
	main()