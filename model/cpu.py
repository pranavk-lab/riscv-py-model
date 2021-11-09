#!/usr/bin/env python3
from numpy.core.numeric import binary_repr, base_repr
from typing import List
from bit_manipulation import BitManip, XLen
from numpy import int32, uint32, uint16, uint8 
from rv32ui import *


class RV32ICORE():
	""" RISC-V 32-bit I-core base class """

	PC: uint32 = uint32(0)
	old_PC: uint32 = uint32(0)
	REG_FILE: List[uint32] = [uint32(0)] * 32
	xlen: int = 32
	instruction_set: dict = {
		0x18 : ConditionalBranch_32,
		0x1b : JumpAndLink_32,
		0x19 : JumpAndLinkRegsiter_32,
		0x0d : LoadUpperImm_32,
		0x05 : AddUpperImmPC_32,
		0x04 : RegImmInt_32,
		0x0C : RegRegInt_32,
		0x00 : Load_32,
		0x08 : Store_32,
		0x03 : Fence_32,
		0x01 : NOP_32
	}

	def __init__(self, mem_size: int = 1024, endianess: str = "little"):
		self.memory = Memory_32(mem_size, endianess)
		self.previous_instruction = NOP_32(0, self)

	def incr_PC(self, factor: int=0x4):
		# Store a temporary PC
		self.old_PC = self.PC

		self.PC+=factor
		if self.PC >= self.memory.mem_size:
			self.PC = uint32(0)

		# Return old PC 
		return self.old_PC
	
	def core_dump(self, mem_dmp="./mem.dump", reg_dmp="./reg.dump"):

		print(f" PC status = {hex(self.old_PC)} -> {hex(self.PC)}")

		self.instr_dump()

		# print(f" Memory dump stored under {mem_dmp}")
		self.memory.mem_dump(mem_dmp)

		# print(f" Register dump stored under {reg_dmp}")
		self.register_dump(reg_dmp)
	

	def register_dump(self, reg_dmp: str):
		with open(reg_dmp, "w") as reg_file:
			reg_file.writelines(
				f"{hex(addr)} | {hex(self.REG_FILE[addr])}\n" 
					for addr in range(32)
			)

	def instr_dump(self):

		instr_fields = self.previous_instruction.dump_instr()

		print("==========================================================")

		for x in range(len(instr_fields)):
			print(f"\t{instr_fields[x][0]}=\t{instr_fields[x][1]}")
		
		print("==========================================================\n")
	
	def st_run(self):

		# single thread pipeline:

		# Fetch -> Decode -> Execute
		self.previous_instruction = self.execute(self.decode(self.fetch()))

		# Write back ???
		# No need to implement a seperate stage for write back in model
		# Execute stratergies can deal with store instructions. 
			
	def fetch(self) -> uint32:
		return uint32(self.memory.read_mem_32(self.PC))
				
	def decode(self, instr : uint32) -> InstructionTemplate_32:

		bm = BitManip(XLen._32BIT)

		opcode1_0, w2 = bm.get_sub_bits_from_instr(instr, 1, 0) 
		opcode6_2, w5 = bm.get_sub_bits_from_instr(instr, 6, 2)

		if  opcode1_0 != 3:
			raise ValueError(f" Not a valid RV32I instruction. \
				instr[1:0] = {binary_repr(opcode1_0, 2)}")

		if opcode6_2 not in self.instruction_set.keys():
			raise ValueError(f" Not a valid RV32I instruction. \
				instr[6:2] = {opcode6_2}")

		return self.instruction_set[opcode6_2](instr, self)
	
	def execute(self, instr: InstructionTemplate_32) -> InstructionTemplate_32:

		if not isinstance(instr, InstructionTemplate_32):
			raise TypeError(f" Invalid instruction type, \
				must be a InstructionTemplate_32 object")

		instr.execute()

		return instr


class Memory_32():
	""" 
	A byte-addressable 32-bit memory model. 
	Provides memory manipulation interface/routines 
	"""

	def __init__(self, mem_size: int = 1024, endianess: str = "little"):

		if endianess == "little":
			self.byte_significance = [4, 3, 2, 1]

		elif endianess == "big":
			self.byte_significance = [1, 2, 3, 4]

		else:
			raise ValueError("Invalid endianess, \
				must be a string matching either 'little' or 'big'")
		
		self.endianess = endianess
		self.mem_size = mem_size
		self.memory = [uint8(0)] * mem_size * 4
	
	def get_bytes_from_word(self, data: uint32) -> tuple:

		bm = BitManip(XLen._32BIT)
		x3_val, w8 = bm.get_sub_bits_from_instr(data, 31, 24)
		x2_val, w8 = bm.get_sub_bits_from_instr(data, 23, 16)
		x1_val, w8 = bm.get_sub_bits_from_instr(data, 15, 8)
		x0_val, w8 = bm.get_sub_bits_from_instr(data, 7, 0)

		# MSB -> LSB
		return (x3_val, x2_val, x1_val, x0_val)

	def write_mem_32(self, addr: int, data: uint32):

		data_bytes_tpl = self.get_bytes_from_word(data)

		for x in range(4):
			self.write_mem_8(
				addr + self.byte_significance[x] - 1, data_bytes_tpl[x]
			)

	def write_mem_16(self, addr: int, data: uint16):

		x3_val, x2_val, msb, lsb = self.get_bytes_from_word(data)

		self.write_mem_8(addr + self.byte_significance[1] % 2, msb)
		self.write_mem_8(addr + self.byte_significance[2] % 2, lsb)

	def write_mem_8(self, addr: int, data: uint8):
		self.memory[addr] = data
			
	def read_mem_32(self, addr: int):

		return BitManip(XLen._32BIT).concat_bits(
			[(self.read_mem_8(addr + self.byte_significance[x] -1), 8) 
				for x in range(4)
			]
		)[0]

	def read_mem_16(self, addr: int) -> uint16:

		bm = BitManip(XLen._32BIT)

		msb = self.read_mem_8(addr + self.byte_significance[1] % 2)
		lsb = self.read_mem_8(addr + self.byte_significance[2] % 2)

		return bm.concat_bits([(msb, 8), (lsb, 8)])[0]

	def read_mem_8(self, addr: int) -> uint8:
		return uint8(self.memory[addr])

	def mem_dump(self, file_name: str):
		with open(file_name, "w") as mem_file:
			mem_file.writelines(
				["{0:5s} : 0x{1:8s}\n".format(
					hex(addr),
					self.hexstr(self.memory[addr+3]) + 
					self.hexstr(self.memory[addr+2]) +
					self.hexstr(self.memory[addr+1]) +
					self.hexstr(self.memory[addr])
				) 
				for addr in range(0, len(self.memory), 4)
			])

	def hexstr(self, val: int) -> str:

		if val == 0:
			return '00'

		return base_repr(val, 16, 2-len(base_repr(val, 16)))
	

class Hex:

	def __init__(self, value : int):
		self.value = hex(value)


if __name__ == "__main__":

	core = RV32ICORE()
	bm = BitManip(XLen._32BIT)
	instr = bm.hex_str_2_unsigned_int("fe20aa23")
	print(f"instruction = {binary_repr(instr, 32)}")

	imm = bm.sign_extend_nbit_2_int32(bm.concat_bits([
		bm.get_sub_bits_from_instr(instr, 31, 25),
		bm.get_sub_bits_from_instr(instr, 11, 7)
	])) 

	src1, w5 = bm.get_sub_bits_from_instr(instr, 19, 15)

	src2, w5 = bm.get_sub_bits_from_instr(instr, 24, 20)

	core.REG_FILE[src1] = 0x2c

	core.REG_FILE[src2] = 0x53a46732

	mem_addr = imm + core.REG_FILE[src1]

	# core.memory.write_mem_32(0x14, 0x53a46732)

	print(f"imm = {int32(imm)}")
	print(f"src1 = {src1}")
	print(f"src2 = {src2}")
	print(f"mem_addr = {hex(mem_addr)}")
	core.PC = 0x4 * 50
	core.memory.write_mem_32(core.PC, instr)
	print(f"PC status = {core.PC}")
	# core.register_dump("input_reg.dump")
	core.core_dump("input_mem.dump", "input_reg.dump")
	# core.memory[0] = instr
	core.st_run()
	print(f"PC status = {core.PC}")
	print(f"memory at {hex(mem_addr)} = {hex(core.memory.read_mem_16(mem_addr))}")
	# core.register_dump("output_reg.dump")
	core.core_dump("output_mem.dump", "output_reg.dump")
	# print(f"result = {int32(core.REG_FILE[dest])}")
