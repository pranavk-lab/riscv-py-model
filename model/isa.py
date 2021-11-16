from abc import ABC, abstractmethod
from enum import Enum
from bit_manipulation import BitManip
import numpy as np

class InstructionType(Enum):
	BASE_INSTR = 0
	DERIVED_INSTR = 1


class InstructionTemplate(ABC):
	""" Template for instrutions """

	@abstractmethod
	def __init__(self, instr, core_state):
		pass

	@abstractmethod
	def execute(self):
		pass

	@abstractmethod
	def dump_instr(self):
		return (())


class InstructionFormats(ABC):
	""" Template for instruction formats/encodings """

	@abstractmethod
	def __init__(self, bm: BitManip, instr: np.uint32) -> None:
		super().__init__()


class ITypeInstr(InstructionFormats):

	def __init__(self, bm: BitManip, instr: np.uint32):
		self.funct3, self.funct3_width = (
			bm.get_sub_bits_from_instr(instr, 14, 12)
		)
		self.imm, self.imm_width = bm.get_sub_bits_from_instr(instr, 31, 20)
		self.shift_imm, self.shift_imm_width = (
			bm.get_sub_bits_from_instr(instr, 31, 25)
		)
		self.shamt, self.shamt_width = (
			bm.get_sub_bits_from_instr(instr, 24, 20)
		)
		self.src, self.src_width = (
			bm.get_sub_bits_from_instr(instr, 19, 15)
		)
		self.dest, self.dest_width = (
			bm.get_sub_bits_from_instr(instr, 11, 7)
		)


class RTypeInstr(InstructionFormats):

	def __init__(self, bm: BitManip, instr: np.uint32):
		self.src2, w5 = bm.get_sub_bits_from_instr(instr, 24, 20)
		self.src1, w5 = bm.get_sub_bits_from_instr(instr, 19, 15)
		self.funct3, w3 = bm.get_sub_bits_from_instr(instr, 14, 12)
		self.funct7, w7 = bm.get_sub_bits_from_instr(instr, 31, 25)
		self.dest, w5 = bm.get_sub_bits_from_instr(instr, 11, 7)

class UTypeInstr(InstructionFormats):

	def __init__(self, bm: BitManip, instr: np.uint32):
		self.u_imm, w32 = bm.concat_bits(
			[bm.get_sub_bits_from_instr(instr, 31, 12), 
			(0, bm.xlen-20)]
		)
		self.dst, w5 = bm.get_sub_bits_from_instr(instr, 11, 7)


class JTypeInstr(InstructionFormats):

	def __init__(self, bm: BitManip, instr: np.uint32):
		self.imm = bm.sign_extend_nbit_2_int(bm.concat_bits([
			bm.get_sub_bits_from_instr(instr, 31, 31), 
			bm.get_sub_bits_from_instr(instr, 19, 12), 
			bm.get_sub_bits_from_instr(instr, 20, 20), 
			bm.get_sub_bits_from_instr(instr, 30, 21) 
		])) 
		self.dst, w5 = bm.get_sub_bits_from_instr(instr, 11, 7)


class BTypeInstr(InstructionFormats):
	
	def __init__(self, bm: BitManip, instr: np.uint32):
		self.funct3, w3 = bm.get_sub_bits_from_instr(instr, 14, 12)
		self.imm, width_offset = bm.concat_bits([
			bm.get_sub_bits_from_instr(instr, 31, 31),
			bm.get_sub_bits_from_instr(instr, 8, 8),
			bm.get_sub_bits_from_instr(instr, 30, 25), 
			bm.get_sub_bits_from_instr(instr, 11, 8)
		])
		self.src1 = bm.get_sub_bits_from_instr(instr, 19, 15)[0]
		self.src2 = bm.get_sub_bits_from_instr(instr, 24, 20)[0] 


class STypeInstr(InstructionFormats):

	def __init__(self, bm: BitManip, instr: np.uint32):
		self.imm = bm.sign_extend_nbit_2_int(bm.concat_bits([
			bm.get_sub_bits_from_instr(instr, 31, 25),
			bm.get_sub_bits_from_instr(instr, 11, 7)
		])) 
		self.src1, w5 = bm.get_sub_bits_from_instr(instr, 19, 15)
		self.src2, w5 = bm.get_sub_bits_from_instr(instr, 24, 20)
		self.funct3, w3 = bm.get_sub_bits_from_instr(instr, 14, 12)
