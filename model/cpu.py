#!/usr/bin/env python3
from numpy.core.numeric import binary_repr
from elf_decode.elf_decode import ELF_DECODE
from typing import List
from abc import ABC, abstractmethod
from bit_manipulation import BitManip_32 as bm

class CPU:
	
	def __init__(self, base_isa: str):
		if base_isa == "rv32i":
			isa = RV32I();

class RV32I:
	""" This class defines the RV32I compatible CPU with built-in memory structure"""

	PC: int = 0
	REG_FILE: List[int] = [0] * 32
	xlen: int = 32

	#TODO: Add a description of this class	
	def __init__(self, mem_size : int):
		self.mem_size = mem_size
		self.address_space = Memory_32(self.mem_size)

	def __incr_PC(self, factor=0x4):
		# Store a temporary PC
		PC = self.PC

		self.PC+=factor
		if self.PC >= self.mem_size:
			self.PC = int(0)

		# Return old PC 
		return PC

	def st_run(self, assm_code, test_cases):
		pass
			
	def __fetcher(self):
		return self.address_space[self.__incr_PC()]
				
	def __decoder(self, inst_hex):
		return RV32I_INSTR(inst_hex)

	def __execute(self, inst_32i):
		instr = inst_32i.bits

		# Conditional Branch
		if inst_32i.opcode == "1100011":
			self.__execute_branch(inst_32i)
		
		# Jump and link register
		elif inst_32i.opcode == "1100111":
			offset = inst_32i.get_int_bits(instr, 31, 20)
			src = inst_32i.get_int_bits(instr, 19, 15)
			dst = inst_32i.get_int_bits(instr, 11, 7)
			self.REG_FILE[dst] = self.PC + 4
			self.PC = offset + src
			
		# Jump and link 
		elif inst_32i.opcode == "1101111":
			offset = int(inst_32i.concat_bits(instr, 31, [19, 12], 20, [30, 21]), 2)
			dst = inst_32i.get_int_bits(instr, 11, 7)
			self.REG_FILE[dst] = self.PC + 4
			self.__incr_PC(offset)

		# Load upper immediate
		elif inst_32i.opcode == "0110111" or inst_32i.opcode == "0010111":
			u_imm = int((inst_32i.get_bits(instr, 31, 12) + 12*"0"), 2)
			dst = inst_32i.get_int_bits(instr, 11, 7)
			if inst_32i.opcode == "0110111":
				self.REG_FILE[dst] = u_imm
			else: 
				self.__incr_PC(u_imm)
		
		# Integer register-immediate instructions
		elif inst_32i.opcode == "0010011":
			self.__execute_int_imm_reg(inst_32i)
		
		# Integer register-register instructions
		elif inst_32i.opcode == "0110011":
			self.__execute_int_reg_reg(inst_32i)

		# Load instructions
		elif inst_32i.opcode == "0000011":
			self.__execute_load(inst_32i)
		
		elif inst_32i.opcode == "0100011":
			self.__execute_store(inst_32i)
			
	def __execute_load(self, inst_32i):
		instr = inst_32i.bits
		imm = inst_32i.get_sign_extended_int_bits(instr, 31, 20, len(instr))
		src1 = inst_32i.get_int_bits(instr, 19, 15)
		dest = inst_32i.get_int_bits(instr, 11, 7)
		funct3 = int(inst_32i.funct3, 2)
		effective_address = self.REG_FILE[src1] + imm

		if funct3 == 0:
			self.REG_FILE[dest] = self.address_space[effective_address]
		
		elif funct3 == 1:
			#TODO: implement a memory class
			pass

	def __execute_store(self, inst_32i):
		#TODO: Finish this routine
		pass
		
	def __execute_int_imm_reg(self, inst_32i):
		instr = inst_32i.bits
		funct3 = int(inst_32i.funct3, 2)
		imm_arith = inst_32i.get_int_bits(instr, 31, 20)
		imm_shift = inst_32i.get_int_bits(instr, 31, 25)
		shamt = inst_32i.get_int_bits(instr, 24, 20)
		src = inst_32i.get_int_bits(instr, 19, 15)
		dest = inst_32i.get_int_bits(instr, 11, 7)

		if funct3 == 0:
			self.REG_FILE[dest] = self.REG_FILE[src] + self.REG_FILE[imm]
			
		elif funct3 == 1:	
			self.REG_FILE[dest] = self.REG_FILE[src] << shamt
		
		elif funct3 == 2:
			self.REG_FILE[dest] = self.REG_FILE[src] 
		#TODO: Finish this routine


	def __execute_int_reg_reg(self, inst_32i):
		instr = inst_32i.bits
		src2 = inst_32i.get_int_bits(instr, 24, 20)
		src1 = inst_32i.get_int_bits(instr, 19, 15)
		dest = inst_32i.get_int_bits(instr, 11, 7)
		#TODO: Finish this routine

	def __execute_branch(self, inst_32i):
		instr = inst_32i.bits
		opcode = inst_32i.opcode
		funct3 = int(inst_32i.funct3, 2)
		offset = inst_32i.concat_bits(instr, 31, 7, [30, 25], [11, 8])
		src1 = int(inst_32i.get_bits(instr, 19, 15), 2)
		src2 = int(inst_32i.get_bits(instr, 24, 20), 2)

		if self.REG_FILE[src1] == self.REG_FILE[src2] and funct3 == 0:
			self.__incr_PC(self.offset)

		elif self.REG_FILE[src1] != self.REG_FILE[src2] and funct3 == 1:
			self.__incr_PC(self.offset)

		elif funct3 == 4 or funct3 == 6:
			if self.REG_FILE[src1] < self.REG_FILE[src2]:
				self.incr_PC(self.offset)
		
		elif funct3 == 5 or funct3 == 7:
			if self.REG_FILE[src1] > self.REG_FILE[src2]:
				self.incr_PC[self.offset]


			
	def __data_memory_access(self, address, data):
		pass
	
	def __write_back(self, address, data):
		pass


class RV32I_INSTR:
	#TODO: Add a description of this class	
	def __init__(self, hex_str):
		self.bm = BitManip()
		self.bits = self.hex2bin(hex_str)
		self.opcodes = self.get_bits(self.bits, 6, 0) 
		self.funct3 = self.get_bits(self.bits, 14, 12)
		self.variant = self.concat_bits(self.bits, [14, 12], [6, 0])

	def concat_bits(self, instr, *args):
		return self.bm.concat_bits(instr, args)
		
	def get_bits(self, instr, upper, lower=None):
		return self.bm.get_bits(instr, upper, lower)

	def get_int_bits(self, instr, upper, lower=None):
		return self.bm.get_int_bits(instr, upper, lower)

	def sign_extend(self, vector, str_len):
		return self.bm.sign_extend(vector, str_len)

	def pad_0s(self, vector, str_len):
		return self.bm.pad_0s(vector, str_len)

	def get_2s_comp(self, vector):
		return self.bm.get_2s_comp(vector)

	def get_sign_extended_int_bits(self, instr, upper, lower, str_len):
		return self.bm.get_sign_extended_int_bits(instr, upper, lower, str_len)

	def get_0s_padded_int_bits(self, instr, upper, lower, str_len):
		return self.bm.get_0s_padded_int_bits(instr, upper, lower, str_len)

	def hex2bin(self, hex_str):
		return self.bm.hex2bin(hex_str)

	def bin2hex(self, vector):
		pass		


class Hex:

	def __init__(self, value : int):
		self.value = hex(value)

class Memory_32:
		#TODO: finish this write-up...
	""" This class is used to model 32-bit addressable memory. It has 
		access to bit manipulation routines via variable "bm". A memory modelA size is specified
		when instantiated. There are 
		three main functions, update_mem_32, update_mem_16, update_mem_8
		provide write functionality on a """	
	def __init__(self, size):
		self.size = size
		self.bm = BitManip()
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

