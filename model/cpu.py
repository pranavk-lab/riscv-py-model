#!/usr/bin/env python3
from enum import Enum
from numpy.core.numeric import binary_repr, base_repr
from typing import List, Tuple
from bit_manipulation import BitManip, XLen
from numpy import uint8, float64
from rv32ui import RV32UI 
from rv64ui import RV64UI
import isa

class Endianess(Enum):
	LITTLE = 0
	BIG = 1

class RISCVCore():
	""" RISC-V I-core class """

	def __init__(
		self, mem_size: int = 1024, isa = RV64UI(),
		endianess = Endianess.LITTLE, xlen = XLen._64BIT):
		""" Initializes CPU parameters """

		self.bm = BitManip(xlen)
		self.isa = isa
		self.memory = Memory(mem_size, endianess, xlen)
		self.PC = isa.uint(0)
		self.previous_instruction = isa.NOP(0, self)
		self.old_PC = isa.uint(0)
		self.REG_FILE_SIZE = 32
		self.REG_FILE = [isa.uint(0)] * self.REG_FILE_SIZE
		self.xlen = xlen
		self.instruction_set = isa.get_instructions()

	def incr_PC(self, factor: int=0x4):
		# Store a temporary PC
		self.old_PC = self.PC

		self.PC+=factor
		if self.PC >= self.memory.mem_size:
			self.PC = self.isa.uint(0)

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
					for addr in range(self.REG_FILE_SIZE)
			)

	def instr_dump(self):

		instr_fields = self.previous_instruction.dump_instr()

		print("==========================================================")

		for x in range(len(instr_fields)):
			print(f"\t{instr_fields[x][0]}=\t{instr_fields[x][1]}")
		
		print("==========================================================\n")
	
	def st_run(self):
		"""single thread pipeline
		No need to implement a seperate stage for write back in model
		Execute stratergies can deal with store instructions. 

		Fetch -> Decode -> Execute
		"""
		self.previous_instruction = self.execute(self.decode(self.fetch()))

	def fetch(self):
		return self.isa.uint(
			self.memory.read_mem_word(self.PC, self.xlen.value//8)
		)
				
	def decode(self, instr) -> isa.InstructionTemplate:

		# print(instr)

		opcode1_0, w2 = self.bm.get_sub_bits_from_instr(instr, 1, 0) 
		opcode6_2, w5 = self.bm.get_sub_bits_from_instr(instr, 6, 2)

		if  opcode1_0 != 3:
			raise ValueError(
				(f" Not a valid {self.isa.name} instruction."), 
				(f"instr[1:0] = {binary_repr(opcode1_0, 2)}")
			)

		if opcode6_2 not in self.instruction_set.keys():
			raise ValueError(
				(f" Not a valid {self.isa.name} instruction."),
				(f"instr[6:2] = {opcode6_2}")
			)

		return self.instruction_set[opcode6_2](instr, self)
	
	def execute(
		self, instr: isa.InstructionTemplate) -> isa.InstructionTemplate:

		if not isinstance(instr, isa.InstructionTemplate):
			raise TypeError(
				(f" Invalid instruction type, "),
				(f"must be a isa.InstructionTemplate object")
			)

		instr.execute()

		return instr

#TODO: Fix 64 bit memory calls. 
class Memory():
	""" 
	A byte-addressable memory model. 
	Provides memory manipulation interface/routines 
	"""

	def __init__(
		self, mem_size: int = 1024, endianess = Endianess.LITTLE, 
		xlen = XLen._64BIT): 

		self.endianess = endianess
		self.bm = BitManip(xlen)
		self.mem_size = mem_size
		self.bytes_per_word = xlen.value//8
		self.memory = [uint8(0)] * self.mem_size * self.bytes_per_word
		self.xlen = xlen
	
	def get_bytes_from_word(self, data, word_len):

		word_bytes = [
			self.bm.get_sub_bits_from_instr(data, x, x-7)[0] 
			for x in range(self.xlen.value-1, -1, -8)
		]

		if self.endianess == Endianess.LITTLE:
			word_bytes.reverse()
			return word_bytes

		# BIG ENDIAN
		return word_bytes[self.bytes_per_word-word_len:self.bytes_per_word]

	def write_mem_word(self, addr: int, data: int, word_len: int):

		if word_len > (self.xlen.value//8):
			raise ValueError(
				f"{word_len} is bigger than xlen/8. xlen = {self.xlen.value}"
			)

		# Split data into bytes, represented as tuple of bytes
		data_bytes_tpl = self.get_bytes_from_word(data, word_len)

		if self.endianess == Endianess.LITTLE:
			[self.write_mem_8(addr + x, data_bytes_tpl[x]) 
				for x in range(word_len)
			]
			return 

		# BIG ENDIAN
		[self.write_mem_8(addr + x, data_bytes_tpl[x]) 
			for x in range(word_len)
		]

	def read_mem_word(self, addr: int, word_len: int) -> int:
		"""Always Reads most significant byte first"""

		if word_len > (self.xlen.value//8):
			raise ValueError(
				f"{word_len} is bigger than xlen/8. xlen = {self.xlen.value}"
			)

		if self.endianess == Endianess.LITTLE:
			return self.bm.concat_bits(
				[(self.read_mem_8(addr + x), 8) 
					for x in range(word_len-1, -1, -1)
				]
			)[0]

		# BIG ENDIAN
		return self.bm.concat_bits(
				[(self.read_mem_8(addr + x), 8) 
					for x in range(word_len)
				]
			)[0]


	def write_mem_64(self, addr: int, data:int):
		self.write_mem_word(addr, data, 8)

	def write_mem_32(self, addr: int, data: int):
		self.write_mem_word(addr, data, 4)

	def write_mem_16(self, addr: int, data: int):
		self.write_mem_word(addr, data, 2)

	def write_mem_8(self, addr: int, data: uint8):
		self.memory[addr] = uint8(data)
	
	def read_stored_word(self, addr: int):
		return self.bm.concat_bits(
			[(self.memory[addr + x], 8) 
				for x in range(self.xlen.value//8)
			]
		)

	def read_mem_64(self, addr: int):
		return self.read_mem_word(addr, 8)

	def read_mem_32(self, addr: int):
		return self.read_mem_word(addr, 4)

	def read_mem_16(self, addr: int):
		return self.read_mem_word(addr, 2)

	def read_mem_8(self, addr: int) -> uint8:
		return uint8(self.memory[int(addr)])

	def mem_dump(self, file_name: str):
		with open(file_name, "w") as mem_file:
			mem_file.writelines(
				[
					self.mem_format(addr)
					for addr in range(0, len(self.memory), self.bytes_per_word)
				]
			)
		
	def mem_format(self, addr: int) -> str:
		data, width = self.read_stored_word(addr)
		hex_data = base_repr(data, self.bm.uint(16))
		offset = self.xlen.value//4 - len(hex_data)
		zero_padded_data = base_repr(
			data, self.bm.uint(16), self.bm.uint(offset)
		)
		return (f"{hex(addr)} : {zero_padded_data}\n")

	def hexstr(self, val: int) -> str:

		if val == 0:
			return '00'

		return base_repr(val, 16, 2-len(base_repr(val, 16)))
	

class Hex:

	def __init__(self, value : int):
		self.value = hex(value)

def mem_test():
	mem = Memory(endianess=Endianess.BIG, xlen=XLen._32BIT)

	data = 0xaa3355ff

	print(base_repr(data, 16, 0))

	mem.write_mem_8(1, data)

	data = mem.read_mem_8(1)

	print(data)

	print(base_repr(int(data), 16, 0))

	mem.mem_dump("./mem.dump")

def cpu_test():
	core = RISCVCore(isa=RV64UI(), xlen=XLen._64BIT)
	bm = BitManip(XLen._64BIT)

	instr = bm.hex_str_2_unsigned_int("f0f0f713")

	PC_test = 0x4 * 20

	# Initialize PC
	core.PC = PC_test

	# Set up src register
	src_val = -5
	core.REG_FILE[1] = src_val

	# upper immediate. pre-calculated
	imm = (bm.sign_extend_nbit_2_unsigned_int((3855, 12)))

	# destination register
	dst = 14

	# Initialize memory with instruction
	core.memory.write_mem_32(core.PC, instr)

	# Single test run
	core.st_run()

	core.core_dump()

	print(imm)

	print(bm.uint(src_val))


def bit_manip_test():
	bm = BitManip(XLen._64BIT)
	data = 0xaa3355ff
	print(bm.get_sub_bits_from_instr(data, 0, 0)[0])

def main():
	cpu_test()
	# mem_test()

if __name__ == "__main__":
	main()