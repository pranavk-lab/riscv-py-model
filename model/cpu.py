#!/usr/bin/env python3
from io import open_code

from numpy.core.numeric import binary_repr
from elf_decode.elf_decode import ELF_DECODE
from typing import List
from abc import ABC, abstractmethod
from bit_manipulation import BitManip32 
from numpy import int32, uint32, uint64, iinfo
from numpy import left_shift, right_shift, bitwise_and, bitwise_xor, bitwise_or

class CPU():
	
	def __init__(self, base_isa: str):
		pass


class InstrExeStratergy(ABC):
	@abstractmethod
	def exe_instr(self, instr: uint32, core_state):
		pass
			

# class Memory32:
# 		#TODO: finish this write-up...
# 	""" This class is used to model 32-bit addressable memory. It has 
# 		access to bit manipulation routines via variable "bm". A memory modelA size is specified
# 		when instantiated. There are 
# 		three main functions, update_mem_32, update_mem_16, update_mem_8
# 		provide write functionality on a """	
# 	def __init__(self, size):
# 		self.size = size
# 		self.bm = bm()
# 		self.address_space = self.size*[0]
# 		self.memory_dump = "mem.dump"

# 	def __update_dump(self):
# 		with open(self.memory_dump, "w") as file_:
# 			file_.write(self.address_space)

# 	#TODO: Finish memory manipulation routines
# 	def update_mem_32(self, address, data):
# 		pass

# 	def update_mem_16(self, address, data):
# 		pass

# 	def update_mem_8(self, address, data):	
# 		if len(data) != 32:
# 			raise ValueError("data passed is not 32 bits")
# 		else:
# 			pass


class RV32ICORE:
	""" This class defines the RV32I compatible CORE """ 

	PC: uint32 = uint32(0)
	REG_FILE: List[uint32] = [uint32(0)] * 32
	xlen: int = 32
	mem_dump_file_name: str = "./mem.dump"
	reg_dump_file_name: str = "./reg.dump"

	def __init__(self, mem_size: int = 1024):
		self.mem_size = mem_size
		self.memory = [0] * mem_size

	def incr_PC(self, factor: int=0x4):
		# Store a temporary PC
		PC = self.PC

		self.PC+=factor
		if self.PC >= self.mem_size:
			self.PC = uint32(0)

		# Return old PC 
		return PC
	
	def core_dump(self, mem_dmp="./mem.dump", reg_dmp="./reg.dump"):

		self.reg_dump_file_name = reg_dmp

		self.mem_dump_file_name = mem_dmp
		
		print(f" PC status = {self.PC}")

		print(f" Memory dump stored under {self.mem_dump_file_name}")

		print(f" Register dump stored under {self.reg_dump_file_name}")

		self.memory_dump()

		self.register_dump()

	def register_dump(self):
		with open(self.reg_dump_file_name, "w") as reg_file:
			for addr in range(len(self.REG_FILE)):
				reg_file.writelines(f"{hex(addr)} | {hex(self.REG_FILE[addr])}\n")

	def memory_dump(self):
		with open(self.mem_dump_file_name, "w") as mem_file:
			for addr in range(len(self.memory)):
				mem_file.writelines(f"{hex(addr)} | {hex(self.memory[addr])}\n")

	def st_run(self):

		# single thread pipeline:

		# Fetch
		instr = self.fetch()

		# Decode
		exe_stratergy = self.decode(instr)()

		# Execute
		self.execute(instr, exe_stratergy)

		# Write back ???
		# No need to implement a seperate stage for write back in model
		# Execute stratergies can deal with store instructions. 
			
	def fetch(self) -> uint32:

		return uint32(self.memory[self.PC])
				
	# TODO: optimize the opcode if-else statements
	def decode(self, instr : uint32) -> InstrExeStratergy:

		bm = BitManip32()

		opcode1_0, w2 = bm.get_sub_bits_from_instr(instr, 1, 0) 

		if  opcode1_0 != 3:
			raise ValueError(f" Not a valid RV32I instruction. instr[1:0] = {binary_repr(opcode1_0, 2)}")
		
		opcode6_2, w5 = bm.get_sub_bits_from_instr(instr, 6, 2)

		if opcode6_2 == 0x18:
			return ConditionalBranch_32
		
		elif opcode6_2 == 0x19:
			return JumpAndLinkRegsiter_32
		
		elif opcode6_2 == 0x1b:
			return JumpAndLink_32
		
		elif opcode6_2 == 0x0d:
			return LoadUpperImm_32
		
		elif opcode6_2 == 0x05:
			return AddUpperImmPC_32
		
		elif opcode6_2 == 0x04:
			return RegImmInt_32

		elif opcode6_2 == 0x0C:
			return RegRegInt_32

		elif opcode6_2 == 0x00:
			return Load_32
		
		elif opcode6_2 == 0x08:
			return Store_32
		
		elif opcode6_2 == 0x03:
			return Fence_32

		else:
			raise ValueError(f" Not a valid RV32I instruction. instr[6:2] = {opcode6_2}")

	def execute(self, instr: uint32, exe_stratergy: InstrExeStratergy):

		# Run exe stratergy
		self = exe_stratergy.exe_instr(instr, self)
		

class ConditionalBranch_32(InstrExeStratergy):
	def exe_instr(self, instr: uint32, core_state: RV32ICORE) -> RV32ICORE:

		bm = BitManip32()

		funct3, w3 = bm.get_sub_bits_from_instr(instr, 14, 12)

		offset, width_offset = bm.concat_bits([
			bm.get_sub_bits_from_instr(instr, 31, 31),
			bm.get_sub_bits_from_instr(instr, 8, 8),
			bm.get_sub_bits_from_instr(instr, 30, 25), 
			bm.get_sub_bits_from_instr(instr, 11, 8)
		])

		src1, w5 = bm.get_sub_bits_from_instr(instr, 19, 15)
		src2, w5 = bm.get_sub_bits_from_instr(instr, 24, 20)

		src1_val = int32(core_state.REG_FILE[src1])
		src2_val = int32(core_state.REG_FILE[src2])

		if funct3 == 0:
			if src1_val == src2_val:
				core_state.incr_PC(offset)
			else:
				core_state.incr_PC()

		elif funct3 == 1:
			if src1_val != src2_val:
				core_state.incr_PC(offset)
			else:
				core_state.incr_PC()

		elif funct3 == 4:
			if src1_val < src2_val:
				core_state.incr_PC(offset)
			else:
				core_state.incr_PC()
		
		elif funct3 == 5:
			if src1_val > src2_val:
				core_state.incr_PC(offset)
			else:
				core_state.incr_PC()
			
		elif funct3 == 6:
			if core_state.REG_FILE[src1] < core_state.REG_FILE[src2]:
				core_state.incr_PC(offset)
			else:
				core_state.incr_PC()

		elif funct3 == 7:
			if core_state.REG_FILE[src1] > core_state.REG_FILE[src2]:
				core_state.incr_PC(offset)
			else:
				core_state.incr_PC()
		
		return core_state


class JumpAndLinkRegsiter_32(InstrExeStratergy):
	def exe_instr(self, instr: uint32, core_state: RV32ICORE) -> RV32ICORE:

		bm = BitManip32()
		
		offset = bm.sign_extend_nbit_2_int32(
			bm.get_sub_bits_from_instr(instr, 31, 20)
		)

		src, w5 = bm.get_sub_bits_from_instr(instr, 19, 15)

		dst, w5 = bm.get_sub_bits_from_instr(instr, 11, 7)

		core_state.REG_FILE[dst] = core_state.PC + 4

		# Add offset and content in REG_FILE[src]
		result = offset + int32(core_state.REG_FILE[src])

		# Set the least significant bit of result to 0. 
		# Don't ask me why? It's in the RISCV specification. 
		core_state.PC = bitwise_and(result, 0xfffffffe)
	
		return core_state
			

class JumpAndLink_32(InstrExeStratergy):
	def exe_instr(self, instr: uint32, core_state: RV32ICORE) -> RV32ICORE:

		bm = BitManip32()

		offset = bm.sign_extend_nbit_2_int32(bm.concat_bits([
			bm.get_sub_bits_from_instr(instr, 31, 31), 
			bm.get_sub_bits_from_instr(instr, 19, 12), 
			bm.get_sub_bits_from_instr(instr, 20, 20), 
			bm.get_sub_bits_from_instr(instr, 30, 21) 
		])) 

		dst, w5 = bm.get_sub_bits_from_instr(instr, 11, 7)

		core_state.REG_FILE[dst] = core_state.PC + 4

		core_state.incr_PC(offset)
	
		return core_state


class LoadUpperImm_32(InstrExeStratergy):
	def exe_instr(self, instr: uint32, core_state: RV32ICORE) -> RV32ICORE:

		bm = BitManip32()

		u_imm, w32 = bm.concat_bits([bm.get_sub_bits_from_instr(instr, 31, 12), (0, 12)])

		dst, w5 = bm.get_sub_bits_from_instr(instr, 11, 7)

		core_state.REG_FILE[dst] = u_imm

		core_state.incr_PC()
		
		return core_state
		

class AddUpperImmPC_32(InstrExeStratergy):
	def exe_instr(self, instr: uint32, core_state: RV32ICORE) -> RV32ICORE:

		bm = BitManip32()

		u_imm, w32 = bm.concat_bits([bm.get_sub_bits_from_instr(instr, 31, 12), (0, 12)])

		dst, w5 = bm.get_sub_bits_from_instr(instr, 11, 7)

		core_state.REG_FILE[dst] = u_imm + core_state.incr_PC()

		return core_state


#TODO: See if you can optimize the if else statements
class RegImmInt_32(InstrExeStratergy):
	def exe_instr(self, instr: uint32, core_state: RV32ICORE) -> RV32ICORE:
		
		bm = BitManip32()
		
		funct3, w3 = bm.get_sub_bits_from_instr(instr, 14, 12)

		imm_arith_v, w12 = bm.get_sub_bits_from_instr(instr, 31, 20)

		imm_shift_v, w7 = bm.get_sub_bits_from_instr(instr, 31, 25)

		unsigned_imm_arith = bm.sign_extend_nbit_2_uint32(imm_arith_v)

		signed_imm_arith = bm.sign_extend_nbit_2_int32(imm_arith_v)

		shamt, w5 = bm.get_sub_bits_from_instr(instr, 24, 20)

		src, w5 = bm.get_sub_bits_from_instr(instr, 19, 15)

		dest, w5 = bm.get_sub_bits_from_instr(instr, 11, 7)

		result = 0

		# ADDI
		if funct3 == 0:
			result = int32(core_state.REG_FILE[src]) + signed_imm_arith
			
		# SLLI
		elif funct3 == 1:	
			result = left_shift(core_state.REG_FILE[src], shamt)
		
		# SLTI
		elif funct3 == 2:
			if core_state.REG_FILE[src] < signed_imm_arith:
				result = 1
			else:
				result = 0

		# SLTIU
		elif funct3 == 3:
			if core_state.REG_FILE[src] < unsigned_imm_arith:
				result = 1
			else:
				result = 0
		
		# XORI
		elif funct3 == 4:
			result = bitwise_xor(core_state.REG_FILE[src], unsigned_imm_arith)

		# SRL | SRA
		elif funct3 == 5:

			# SRA
			if imm_shift_v == 32:
				result = right_shift(int32(core_state.REG_FILE[src]), shamt)
			
			# SRL
			elif imm_shift_v == 0:
				result = right_shift(core_state.REG_FILE[src], shamt)
			
			else:
				raise ValueError(f" immediate[11:5] needs to be either 32 or 0. Actual imm = {imm_shift_v}")
		
		# OR
		elif funct3 == 6:
			result = bitwise_or(core_state.REG_FILE[src], unsigned_imm_arith)
		
		# AND
		elif funct3 == 7:
			result = bitwise_and(core_state.REG_FILE[src], unsigned_imm_arith)
		
		core_state.REG_FILE[dest] = uint32(result)

		core_state.incr_PC()

		return core_state


class RegRegInt_32(InstrExeStratergy):
	def exe_instr(self, instr: uint32, core_state: RV32ICORE) -> RV32ICORE:

		bm = BitManip32()

		src2, w5 = bm.get_sub_bits_from_instr(instr, 24, 20)

		src1, w5 = bm.get_sub_bits_from_instr(instr, 19, 15)

		funct3, w3 = bm.get_sub_bits_from_instr(instr, 14, 12)

		funct7, w7 = bm.get_sub_bits_from_instr(instr, 31, 25)

		dest, w5 = bm.get_sub_bits_from_instr(instr, 11, 7)

		src1_val = core_state.REG_FILE[src1]

		src2_val = core_state.REG_FILE[src2]

		result = 0

		# ADD | SUB
		if funct3 == 0:

			# ADD
			if funct7 == 0:
				result = src1_val + src2_val

			# SUB
			elif funct7 == 32:
				result = src1_val - src2_val

			else:
				raise ValueError(f"funct7 needs to be either 32 or 0. Actual funct7 = {funct7}")
			
		# SLL
		elif funct3 == 1:	
			result = left_shift(src1_val, bm.get_sub_bits_from_instr(src2_val, 4, 0))
		
		# SLT
		elif funct3 == 2:
			if int32(src1_val) < int32(src2_val):
				result = 1
			else:
				result = 0

		# SLTIU
		elif funct3 == 3:
			if src1_val < src2_val:
				result = 1
			else:
				result = 0
		
		# XORI
		elif funct3 == 4:
			result = bitwise_xor(src1_val, src2_val)

		# SRL | SRA
		elif funct3 == 5:

			# SRA
			if funct7 == 32:
				result = right_shift(int32(src1_val), bm.get_sub_bits_from_instr(src2_val, 4, 0))
			
			# SRL
			elif funct7 == 0:
				result = right_shift(src1_val, bm.get_sub_bits_from_instr(src2_val, 4, 0))
			
			else:
				raise ValueError(f" funct7 needs to be either 32 or 0. Actual funct7 = {funct7}")
		
		# OR
		elif funct3 == 6:
			result = bitwise_or(src1_val, src2_val)
		
		# AND
		elif funct3 == 7:
			result = bitwise_and(src1_val, src2_val)
		
		core_state.REG_FILE[dest] = uint32(result)

		core_state.incr_PC()

		return core_state


class Load_32(InstrExeStratergy):
	def exe_instr(self, instr: uint32, core_state: RV32ICORE) -> RV32ICORE:

		bm = BitManip32()

		imm, w12 = bm.get_sub_bits_from_instr(instr, 31, 20)

		src, w5 = bm.get_sub_bits_from_instr(instr, 19, 15)

		dest, w5 = bm.get_sub_bits_from_instr(instr, 11, 7)

		funct3, w3 = bm.get_sub_bits_from_instr(instr, 14, 12)

		mem_addr = core_state.REG_FILE[src] + bm.sign_extend_nbit_2_int32((imm, w12))

		mem_data = 0

		# signed loads
		if funct3 <= 2:
			mem_data = bm.sign_extend_nbit_2_int32(core_state.memory[mem_addr])
		
		# unsigned loads
		elif funct3 > 2 and funct3 < 6:
			mem_data = bm.sign_extend_nbit_2_uint32(core_state.memory[mem_addr])
				
		else:
			raise ValueError(f"funct3 is out of scope. Must be 0 <= funct3 <= 5. Actual funct3 = {funct3}")

		core_state.REG_FILE[dest] = uint32(mem_data)

		core_state.incr_PC()
	
		return core_state
		

class Store_32(InstrExeStratergy):
	def exe_instr(self, instr: uint32, core_state: RV32ICORE) -> RV32ICORE:
	
		bm = BitManip32()

		imm, w12 = bm.concat_bits([
			bm.get_sub_bits_from_instr(instr, 31, 25),
			bm.get_sub_bits_from_instr(instr, 14, 12)
		]) 

		src1, w5 = bm.get_sub_bits_from_instr(instr, 19, 15)

		src2, w5 = bm.get_sub_bits_from_instr(instr, 24, 20)

		funct3, w3 = bm.get_sub_bits_from_instr(instr, 14, 12)

		mem_addr = core_state.REG_FILE[src1] + bm.sign_extend_nbit_2_int32((imm, w12))

		# byte store
		if funct3 == 0:
			upper = 7
		
		# 2 byte store
		elif funct3 == 1:
			upper = 15
		
		# 32 bit store
		elif funct3 == 2:
			upper = 32
				
		else:
			raise ValueError(f"funct3 is out of scope. Must be 0 <= funct3 <= 2. Actual funct3 = {funct3}")
		
		src2_tpl = bm.get_sub_bits_from_instr(core_state.REG_FILE[src2], upper, 0)

		core_state.memory[mem_addr] = bm.sign_extend_nbit_2_uint32(src2_tpl)

		core_state.incr_PC()
	
		return core_state


class Fence_32(InstrExeStratergy):
	def exe_instr(self, instr: uint32, core_state: RV32ICORE) -> RV32ICORE:

		bm = BitManip32()

		#TODO: finish this routine
		return core_state	

class Hex:

	def __init__(self, value : int):
		self.value = hex(value)


if __name__ == "__main__":

	core = RV32ICORE()
	bm = BitManip32()
	instr = bm.hex_str_2_unsigned_int("000302e7")
	print(f"instruction = {binary_repr(instr, 32)}")

	offset = bm.sign_extend_nbit_2_int32(
		bm.get_sub_bits_from_instr(instr, 31, 20)
	)
	src, w5 = bm.get_sub_bits_from_instr(instr, 19, 15)
	dst, w5 = bm.get_sub_bits_from_instr(instr, 11, 7)
	funct3, w3 = bm.get_sub_bits_from_instr(instr, 14, 12)
	src1, w5 = bm.get_sub_bits_from_instr(instr, 19, 15)
	src2, w5 = bm.get_sub_bits_from_instr(instr, 24, 20)
	print(f"offset = {offset}")
	# print(f"src1 = {src1}")
	# print(f"src2 = {src2}")
	# print(f"funct3 = {funct3}")
	print(f"src = {src}")
	print(f"dest = {dst}")
	core.core_dump("input_mem.dump", "input_reg.dump")
	core.memory[0] = instr
	core.st_run()
	core.core_dump("output_mem.dump", "output_reg.dump")
