#!/usr/bin/env python3
from elf_decode.elf_decode import ELF_DECODE
from typing import List
from abc import ABC, abstractmethod
from bit_manipulation import BitManip32 as bm
from numpy import uint32, uint64

class CPU():
	
	def __init__(self, base_isa: str):
		pass


class InstrExeStratergy32(ABC):
	@abstractmethod
	def exe_instr(self, instr: uint32, core_state):
		pass
			

class Memory32:
		#TODO: finish this write-up...
	""" This class is used to model 32-bit addressable memory. It has 
		access to bit manipulation routines via variable "bm". A memory modelA size is specified
		when instantiated. There are 
		three main functions, update_mem_32, update_mem_16, update_mem_8
		provide write functionality on a """	
	def __init__(self, size):
		self.size = size
		self.bm = bm()
		self.address_space = self.size*[0]
		self.memory_dump = "mem.dump"

	def __update_dump(self):
		with open(self.memory_dump, "w") as file_:
			file_.write(self.address_space)

	#TODO: Finish memory manipulation routines
	def update_mem_32(self, address, data):
		pass

	def update_mem_16(self, address, data):
		pass

	def update_mem_8(self, address, data):	
		if len(data) != 32:
			raise ValueError("data passed is not 32 bits")
		else:
			pass


class RV32ICORE:
	""" This class defines the RV32I compatible CORE """ 

	PC: int = 0
	REG_FILE: List[int] = [0] * 32
	xlen: int = 32

	def __init__(self, memory: Memory32):
		self.memory = memory

	def __incr_PC(self, factor: int=0x4):
		# Store a temporary PC
		PC = self.PC

		self.PC+=factor
		if self.PC >= self.mem_size:
			self.PC = int(0)

		# Return old PC 
		return PC

	def st_run(self):

		# single thread pipeline:

		# Fetch
		instr = self.__fetch

		# Decode
		exe_stratergy = self.__decode(instr)

		# Execute
		self.__execute(instr, exe_stratergy)

		# Write back ???
		# No need to implement a seperate stage for write back in model
		# Execute stratergies can deal with store instructions. 
			
	def __fetch(self) -> uint32:

		# Get 32 bit data in memory[PC], then PC++
		return self.address_space[self.__incr_PC]
				
	def __decode(self, instr : uint32) -> InstrExeStratergy32:
		#TODO: finish this function
		pass

	def __execute(self, instr: uint32, exe_stratergy: InstrExeStratergy32):

		# Run exe stratergy
		self = exe_stratergy.exe_instr(instr, self)
		

class ConditionalBranch_32(InstrExeStratergy32):
	def exe_instr(self, instr: uint32, core_state: RV32ICORE) -> RV32ICORE:

		opcode, w7 = bm.get_sub_bits_from_instr(instr, 6, 0)

		funct3, w3 = bm.get_sub_bits_from_instr(instr, 14, 12)

		offset, width_offset = bm.concat_bits([
			bm.get_sub_bits_from_instr(instr, 31, 31),
			bm.get_sub_bits_from_instr(instr, 8, 8),
			bm.get_sub_bits_from_instr(instr, 30, 25), 
			bm.get_sub_bits_from_instr(instr, 11, 8)
		])

		src1, w5 = bm.get_sub_bits_from_instr(instr, 19, 15)
		src2, w5 = bm.get_sub_bits_from_instr(instr, 24, 20)

		if core_state.REG_FILE[src1] == core_state.REG_FILE[src2] and funct3 == 0:
			core_state.__incr_PC(offset)

		elif core_state.REG_FILE[src1] != core_state.REG_FILE[src2] and funct3 == 1:
			core_state.__incr_PC(offset)

		elif funct3 == 4 or funct3 == 6:
			if core_state.REG_FILE[src1] < core_state.REG_FILE[src2]:
				core_state.incr_PC(offset)
		
		elif funct3 == 5 or funct3 == 7:
			if core_state.REG_FILE[src1] > core_state.REG_FILE[src2]:
				core_state.incr_PC[offset]
		
		return core_state


class JumpAndLinkRegsiter_32(InstrExeStratergy32):
	def exe_instr(self, instr: uint32, core_state: RV32ICORE) -> RV32ICORE:
		
		offset, w12 = bm.get_sub_bits_from_instr(instr, 31, 20)

		src, w5 = bm.get_sub_bits_from_instr(instr, 19, 15)

		dst, w5 = bm.get_sub_bits_from_instr(instr, 11, 7)

		core_state.REG_FILE[dst] = core_state.PC + 4

		core_state.PC = offset + src
	
		return core_state
			

class JumpAndLink_32(InstrExeStratergy32):
	def exe_instr(self, instr: uint32, core_state: RV32ICORE) -> RV32ICORE:
		# elif bm.opcode == "1101111":

		offset, w20 = bm.get_sub_bits_from_instr([
			bm.get_sub_bits_from_instr(instr, 31), 
			bm.get_sub_bits_from_instr(instr, 19, 12), 
			bm.get_sub_bits_from_instr(instr, 20), 
			bm.get_sub_bits_from_instr(instr, 30, 21), 
			bm.get_sub_bits_from_instr(instr, 2)	
		]) 

		dst, w5 = bm.get_sub_bits_from_instr(instr, 11, 7)

		core_state.REG_FILE[dst] = core_state.PC + 4

		core_state.__incr_PC(offset)
	
		return core_state


class LoadUpperImm_32(InstrExeStratergy32):
	def exe_instr(self, instr: uint32, core_state: RV32ICORE) -> RV32ICORE:
		# elif bm.opcode == "0110111" or bm.opcode == "0010111":

		u_imm = bm.concat_bits([bm.get_sub_bits_from_instr(instr, 31, 12), (0, 12)])[0]

		dst = bm.get_sub_bits_from_instr(instr, 11, 7)[0]

		if bm.opcode == "0110111":
			core_state.REG_FILE[dst] = u_imm
		else: 
			core_state.__incr_PC(u_imm)
		
		return core_state
		

class RegImmInt_32(InstrExeStratergy32):
	def exe_instr(self, instr: uint32, core_state: RV32ICORE) -> RV32ICORE:
		# elif bm.opcode == "0010011":
		
		funct3, w3 = bm.get_sub_bits_from_instr(instr, 14, 12)

		imm_arith, w12 = bm.get_sub_bits_from_instr(instr, 31, 20)

		imm_shift, w7 = bm.get_sub_bits_from_instr(instr, 31, 25)

		shamt, w5 = bm.get_sub_bits_from_instr(instr, 24, 20)

		src, w5 = bm.get_sub_bits_from_instr(instr, 19, 15)

		dest, w5 = bm.get_sub_bits_from_instr(instr, 11, 7)

		if funct3 == 0:
			core_state.REG_FILE[dest] = core_state.REG_FILE[src] + core_state.REG_FILE[imm_arith]
			
		elif funct3 == 1:	
			core_state.REG_FILE[dest] = core_state.REG_FILE[src] << shamt
		
		elif funct3 == 2:
			core_state.REG_FILE[dest] = core_state.REG_FILE[src] 
		#TODO: Finish this routine
	
		return core_state


class RegRegInt_32(InstrExeStratergy32):
	def exe_instr(self, instr: uint32, core_state: RV32ICORE) -> RV32ICORE:
		# elif bm.opcode == "0110011":
		src2, w5 = bm.get_sub_bits_from_instr(instr, 24, 20)
		src1, w5 = bm.get_sub_bits_from_instr(instr, 19, 15)
		dest, w5 = bm.get_sub_bits_from_instr(instr, 11, 7)
		#TODO: Finish this routine
	
		return core_state


class Load_32(InstrExeStratergy32):
	#TODO: finish this routine
	def exe_instr(self, instr: uint32, core_state: RV32ICORE) -> RV32ICORE:
		# elif bm.opcode == "0000011":
		imm = bm.sign_extend_nbit_2_32bit(bm.get_sub_bits_from_instr(instr, 31, 20))
		src1, w5 = bm.get_sub_bits_from_instr(instr, 19, 15)
		dest, w5 = bm.get_sub_bits_from_instr(instr, 11, 7)
		funct3, w3 = bm.get_sub_bits_from_instr(instr, 14, 12)
		effective_address = core_state.REG_FILE[src1] + imm

		if funct3 == 0:
			pass
			# core_state.REG_FILE[dest] = core_state.address_space[effective_address]
		
		elif funct3 == 1:
			#TODO: implement a memory class
			pass
	
		return core_state
		

class Store_32(InstrExeStratergy32):
	def exe_instr(self, instr: uint32, core_state: RV32ICORE) -> RV32ICORE:
		# elif bm.opcode == "0100011":
	
		return core_state


class Hex:

	def __init__(self, value : int):
		self.value = hex(value)


