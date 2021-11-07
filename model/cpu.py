#!/usr/bin/env python3
from numpy.core.numeric import binary_repr, base_repr
from typing import List
from abc import ABC, abstractmethod
from bit_manipulation import BitManip32 
from numpy import add, int32, uint, uint32, uint16, uint8 
from numpy import left_shift, right_shift, bitwise_and, bitwise_xor, bitwise_or


class RISCVCORE_TYPE():
	""" Defines risc-v core type hint """


class InstructionTemplate_32(ABC):
	""" Template for 32-bit instrutions """

	@abstractmethod
	def __init__(self, instr: uint32, core_state: RISCVCORE_TYPE):
		pass

	@abstractmethod
	def execute(self):
		pass

	@abstractmethod
	def dump_instr(self) -> tuple:
		return (())

class ConditionalBranch_32(InstructionTemplate_32):

	def beq(self):
		if int32(self.src1_val) == int32(self.src2_val):
			self.core_state.incr_PC(self.offset)
		else:
			self.core_state.incr_PC()
	
	def bne(self):
		if int32(self.src1_val) != int32(self.src2_val):
			self.core_state.incr_PC(self.offset)
		else:
			self.core_state.incr_PC()
		
	def blt(self):
		if int32(self.src1_val) < int32(self.src2_val):
			self.core_state.incr_PC(self.offset)
		else:
			self.core_state.incr_PC()
	
	def bge(self):
		if int32(self.src1_val) > int32(self.src2_val):
			self.core_state.incr_PC(self.offset)
		else:
			self.core_state.incr_PC()

	def bltu(self):
		if self.src1_val < self.src2_val:
			self.core_state.incr_PC(self.offset)
		else:
			self.core_state.incr_PC()
	
	def bgeu(self):
		if self.src1_val > self.src2_val:
			self.core_state.incr_PC(self.offset)
		else:
			self.core_state.incr_PC()

	def __init__(self, instr: uint32, core_state: RISCVCORE_TYPE):

		bm = BitManip32()
		self.branch_function = {
			0 : self.beq,
			1 : self.bne,
			4 : self.blt,
			5 : self.bge,
			6 : self.bltu,
			7 : self.bgeu 
		}
		self.core_state = core_state
		self.PC_state = self.core_state.PC
		self.funct3, w3 = bm.get_sub_bits_from_instr(instr, 14, 12)
		self.offset, width_offset = bm.concat_bits([
			bm.get_sub_bits_from_instr(instr, 31, 31),
			bm.get_sub_bits_from_instr(instr, 8, 8),
			bm.get_sub_bits_from_instr(instr, 30, 25), 
			bm.get_sub_bits_from_instr(instr, 11, 8)
		])
		self.src1 = bm.get_sub_bits_from_instr(instr, 19, 15)[0]
		self.src1_val = core_state.REG_FILE[self.src1]

		self.src2 = bm.get_sub_bits_from_instr(instr, 24, 20)[0] 
		self.src2_val = core_state.REG_FILE[self.src2]
	
	def execute(self):

		if self.funct3 not in self.branch_function.keys():
			raise ValueError(f" Invalid funct3 = {self.funct3}")

		self.branch_function[self.funct3]()
	
	def dump_instr(self) -> tuple:
		return (
			("instr", "branch instruction"),
			("funct3", self.funct3),
			("offset", self.offset),
			("src1", self.src1),
			("src1_val", self.src1_val),
			("src2", self.src2_val)
		)
	

class JumpAndLinkRegsiter_32(InstructionTemplate_32):

	def __init__(self, instr: uint32, core_state: RISCVCORE_TYPE):

		bm = BitManip32()
		self.core_state = core_state

		self.offset = bm.sign_extend_nbit_2_int32(
			bm.get_sub_bits_from_instr(instr, 31, 20)
		)

		self.src, w5 = bm.get_sub_bits_from_instr(instr, 19, 15)

		self.dst, w5 = bm.get_sub_bits_from_instr(instr, 11, 7)

		self.src1_val = core_state.REG_FILE[self.src]

		self.PC_state = self.core_state.PC

	def execute(self):
		self.core_state.REG_FILE[self.dst] = self.core_state.PC + 4

		# Add offset and content in REG_FILE[src]
		result = self.offset + int32(self.core_state.REG_FILE[self.src])

		# Set the least significant bit of result to 0. 
		# Don't ask me why? It's in the RISCV specification. 
		self.core_state.PC = bitwise_and(result, 0xfffffffe)
	
	def dump_instr(self) -> tuple:
		return (
			("instr", "JALR instruction"),
			("offset", self.offset),
			("src", self.src),
			("src_val", self.src1_val),
			("dest", self.dst)
		)


class JumpAndLink_32(InstructionTemplate_32):

	def __init__(self, instr: uint32, core_state: RISCVCORE_TYPE):
		bm = BitManip32()

		self.offset = bm.sign_extend_nbit_2_int32(bm.concat_bits([
			bm.get_sub_bits_from_instr(instr, 31, 31), 
			bm.get_sub_bits_from_instr(instr, 19, 12), 
			bm.get_sub_bits_from_instr(instr, 20, 20), 
			bm.get_sub_bits_from_instr(instr, 30, 21) 
		])) 

		self.dst, w5 = bm.get_sub_bits_from_instr(instr, 11, 7)

		self.core_state = core_state

	def execute(self):

		self.core_state.REG_FILE[self.dst] = self.core_state.PC + 4

		self.core_state.incr_PC(self.offset)

	def dump_instr(self) -> tuple:
		return (
			("instr", "JAL instruction"),
			("offset", self.offset),
			("dest", self.dst)
		)

class LoadUpperImm_32(InstructionTemplate_32):

	def __init__(self, instr: uint32, core_state: RISCVCORE_TYPE):
		bm = BitManip32()

		self.u_imm, w32 = bm.concat_bits(
			[bm.get_sub_bits_from_instr(instr, 31, 12), (0, 12)]
		)

		self.dst, w5 = bm.get_sub_bits_from_instr(instr, 11, 7)

		self.core_state = core_state

	def execute(self):
		self.core_state.REG_FILE[self.dst] = self.u_imm

		self.core_state.incr_PC()
	
	def dump_instr(self) -> tuple:
		return (
			("instr", "LUI instruction"),
			("imm", self.u_imm),
			("dst", self.dst),
		)

class AddUpperImmPC_32(InstructionTemplate_32): 

	def __init__(self, instr: uint32, core_state: RISCVCORE_TYPE):
		bm = BitManip32()

		self.u_imm, w32 = bm.concat_bits(
			[bm.get_sub_bits_from_instr(instr, 31, 12), (0, 12)]
		)
		self.dst, w5 = bm.get_sub_bits_from_instr(instr, 11, 7)
		self.core_state = core_state

	def execute(self):

		self.core_state.REG_FILE[self.dst] = (
			self.u_imm + self.core_state.incr_PC()
		)

	def dump_instr(self) -> tuple:
		return (
			("instr", "AUI instruction"),
			("imm", self.u_imm),
			("dst", self.dst),
		)


class RegImmInt_32(InstructionTemplate_32):

	def addi(self) -> uint32:
		return int32(self.src_val) + self.signed_imm_arith
	
	def slli(self) -> uint32:
		return left_shift(self.src_val, self.shamt)

	def slti(self) -> uint32:
		if int32(self.src_val) < self.signed_imm_arith:
			return 1
		else:
			return 0

	def sltiu(self) -> uint32:
		if self.src_val < self.unsigned_imm_arith:
			return 1
		else:
			return 0
		
	def xori(self) -> uint32:
		return bitwise_xor(self.src_val, self.unsigned_imm_arith)

	def srai_srli(self) -> uint32:
		# SRA
		if self.imm_shift_v == 32:
			return right_shift(int32(self.src_val), self.shamt)
		
		# SRL
		elif self.imm_shift_v == 0:
			return right_shift(self.src_val, self.shamt)
		
		else:
			raise ValueError(f" immediate[11:5] needs to be either 32 or 0. \
				Actual imm = {self.imm_shift_v}")
	
	def ori(self) -> uint32:
		return bitwise_or(self.src_val, self.unsigned_imm_arith)
		
	def andi(self) -> uint32:
		return bitwise_and(self.src_val, self.unsigned_imm_arith)

	def __init__(self, instr: uint32, core_state: RISCVCORE_TYPE) -> RISCVCORE_TYPE:
		bm = BitManip32()

		# funct3 dict
		self.int_function = {
			0 : self.addi,
			1 : self.slli,
			2 : self.slti,
			3 : self.sltiu,
			4 : self.xori,
			5 : self.srai_srli,
			6 : self.ori,
			7 : self.andi
		}
		
		# Get all the instruction vectors
		self.funct3, w3 = bm.get_sub_bits_from_instr(instr, 14, 12)
		self.imm_arith_v = bm.get_sub_bits_from_instr(instr, 31, 20)
		self.imm_shift_v, w7 = bm.get_sub_bits_from_instr(instr, 31, 25)
		self.unsigned_imm_arith = bm.sign_extend_nbit_2_uint32(self.imm_arith_v)
		self.signed_imm_arith = bm.sign_extend_nbit_2_int32(self.imm_arith_v)
		self.shamt, w5 = bm.get_sub_bits_from_instr(instr, 24, 20)
		self.src, w5 = bm.get_sub_bits_from_instr(instr, 19, 15)
		self.dest, w5 = bm.get_sub_bits_from_instr(instr, 11, 7)
		self.src_val = uint32(core_state.REG_FILE[self.src])
		self.core_state = core_state

	def execute(self):
		# Execute instruction based on funct3
		self.core_state.REG_FILE[self.dest] = (
			uint32(self.int_function[self.funct3]())
		)

		self.core_state.incr_PC()
	
	def dump_instr(self) -> tuple:
		if self.funct3 == 1 or self.funct3 == 5:
			return (
				("instr", "shift instruction"),
				("funct3", self.funct3),
				("src1", self.src),
				("src1_val", self.src_val),
				("dest", self.dest),
				("shift", self.shamt),
				("imm", self.imm_shift_v)
			)

		return (
			("instr", "ADDI/SLT/LOGICAL instruction"),
			("funct3", self.funct3),
			("src1", self.src),
			("src1_val", self.src_val),
			("dest", self.dest),
			("imm", self.imm_arith_v)
		)

class RegRegInt_32(InstructionTemplate_32):

	def add_sub(self) -> uint32:

		# ADD
		if self.funct7 == 0:
			return int32(self.src1_val) + int32(self.src2_val)

		# SUB
		elif self.funct7 == 32:
			return int32(self.src1_val) - int32(self.src2_val)

		else:
			raise ValueError(f"funct7 needs to be either 32 or 0. \
				Actual funct7 = {self.funct7}")
		
	def sll(self) -> uint32:
		return left_shift(self.src1_val, self.shift_val)
	
	def slt(self) -> uint32:
		if int32(self.src1_val) < int32(self.src2_val):
			return 1
		else:
			return 0

	def sltu(self) -> uint32:
		if self.src1_val < self.src2_val:
			return 1
		else:
			return 0
	
	def xor_(self) -> uint32:
		return bitwise_xor(self.src1_val, self.src2_val)

	def sra_srl(self) -> uint32:

		if self.funct7 == 32:
			return right_shift(int32(self.src1_val), self.shift_val)
		
		elif self.funct7 == 0:
			return right_shift(self.src1_val, self.shift_val)
		
		else:
			raise ValueError(f" funct7 needs to be either 32 or 0. \
			Actual funct7 = {self.funct7}")
	
	def or_(self) -> uint32:
		return bitwise_or(self.src1_val, self.src2_val)
	
	def and_(self) -> uint32:
		return bitwise_and(self.src1_val, self.src2_val)

	def __init__(self, instr: uint32, core_state: RISCVCORE_TYPE):
		bm = BitManip32()

		self.int_function = {
			0 : self.add_sub,
			1 : self.sll,
			2 : self.slt,
			3 : self.sltu,
			4 : self.xor_,
			5 : self.sra_srl,
			6 : self.or_,
			7 : self.and_
		}

		self.src2, w5 = bm.get_sub_bits_from_instr(instr, 24, 20)

		self.src1, w5 = bm.get_sub_bits_from_instr(instr, 19, 15)

		self.funct3, w3 = bm.get_sub_bits_from_instr(instr, 14, 12)

		self.funct7, w7 = bm.get_sub_bits_from_instr(instr, 31, 25)

		self.dest, w5 = bm.get_sub_bits_from_instr(instr, 11, 7)

		self.src1_val = uint32(core_state.REG_FILE[self.src1])

		self.src2_val = uint32(core_state.REG_FILE[self.src2])

		self.shift_val, w5 = bm.get_sub_bits_from_instr(self.src2_val, 4, 0)

		self.core_state = core_state

	def execute(self):
		self.core_state.REG_FILE[self.dest] = (
			uint32(self.int_function[self.funct3]())
		)
		self.core_state.incr_PC()
	
	def dump_instr(self) -> tuple:
		return (
			("instr", "Reg Reg int instruction"),
			("funct3", self.funct3),
			("funct7", self.funct7),
			("src1", self.src1),
			("src1_val", self.src1_val),
			("src2", self.src2),
			("src2_val", self.src2_val),
			("dest", self.dest),
		)

class Load_32(InstructionTemplate_32):

	def lb(self) -> uint32:
		return self.bm.sign_extend_nbit_2_uint32((self.lbu(), 8))

	def lh(self) -> uint32:
		return self.bm.sign_extend_nbit_2_uint32((self.lhu(), 16))
	
	def lw(self) -> uint32:
		return self.core_state.memory.read_mem_32(self.mem_addr)
	
	def lhu(self) -> uint32:
		return self.core_state.memory.read_mem_16(self.mem_addr)
	
	def lbu(self) -> uint32:
		return self.core_state.memory.read_mem_8(self.mem_addr)

	def __init__(self, instr: uint32, core_state: RISCVCORE_TYPE):

		self.bm = BitManip32()

		self.core_state = core_state

		imm, w12 = self.bm.get_sub_bits_from_instr(instr, 31, 20)
		self.src, w5 = self.bm.get_sub_bits_from_instr(instr, 19, 15)
		self.dest, w5 = self.bm.get_sub_bits_from_instr(instr, 11, 7)
		self.funct3, w3 = self.bm.get_sub_bits_from_instr(instr, 14, 12)
		self.mem_addr = (
			core_state.REG_FILE[self.src] + 
			self.bm.sign_extend_nbit_2_int32((imm, w12))
		)
		self.load_function = {
			0 : self.lb,
			1 : self.lh,
			2 : self.lw,
			4 : self.lbu,
			5 : self.lhu 
		}

	def execute(self):
		if self.funct3 not in self.load_function.keys():
			raise ValueError(f" Invalid funct3. \
				Must be {self.load_function.keys()}.\
					Actual funct3 = {self.funct3}")

		self.core_state.REG_FILE[self.dest] = (
			uint32(self.load_function[self.funct3]())
		)
		self.core_state.incr_PC()
	
	def dump_instr(self) -> tuple:
		return (
			("instr", "load instruction"),
			("funct3", self.funct3),
			("src1", self.src),
			("mem_addr", self.mem_addr),
			("dest", self.dest),
		)


class Store_32(InstructionTemplate_32):

	def sb(self):
		self.core_state.memory.write_mem_8(self.mem_addr, uint8(self.src2_val))

	def sh(self):
		self.core_state.memory.write_mem_16(self.mem_addr, uint16(self.src2_val))

	def sw(self):
		self.core_state.memory.write_mem_32(self.mem_addr, uint32(self.src2_val))

	def __init__(self, instr: uint32, core_state: RISCVCORE_TYPE):
		bm = BitManip32()

		imm = bm.sign_extend_nbit_2_int32(bm.concat_bits([
			bm.get_sub_bits_from_instr(instr, 31, 25),
			bm.get_sub_bits_from_instr(instr, 11, 7)
		])) 

		self.core_state = core_state

		self.src1, w5 = bm.get_sub_bits_from_instr(instr, 19, 15)

		self.src2, w5 = bm.get_sub_bits_from_instr(instr, 24, 20)

		self.funct3, w3 = bm.get_sub_bits_from_instr(instr, 14, 12)

		self.src1_val = core_state.REG_FILE[self.src1]

		self.src2_val = core_state.REG_FILE[self.src2]

		self.mem_addr = imm + self.src1_val

		self.store_function = {
			0 : self.sb,
			1 : self.sh,
			2 : self.sw
		}

	def execute(self):
				
		if self.funct3 not in self.store_function.keys():
			raise ValueError(f"funct3 is out of scope. \
				Must be 0 <= funct3 <= 2. Actual funct3 = {self.funct3}")

		# Run store instruction
		self.store_function[self.funct3]()
		
		self.core_state.incr_PC()
	
	def dump_instr(self) -> tuple:
		return (
			("instr", "store instruction"),
			("funct3", self.funct3),
			("src1", self.src1),
			("src1_val", self.src1_val),
			("src2", self.src2),
			("src2_val", self.src2_val),
			("mem_addr", self.mem_addr),
		)


class NOP_32(InstructionTemplate_32):

	def __init__(self, instr: uint32, core_state: RISCVCORE_TYPE):
		self.PC_state = core_state.PC
	
	def execute(self):
		return super().execute()
	
	def dump_instr(self) -> tuple:
		return super().dump_instr()

class Fence_32(InstructionTemplate_32):
	def __init__(instr: uint32, core_state: RISCVCORE_TYPE):
		bm = BitManip32()

		#TODO: finish this routine
		return core_state	

	def execute(self):
		pass


class RV32ICORE():
	""" RISC-V 32-bit core base class """

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

		bm = BitManip32()

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

		bm = BitManip32()
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

		return BitManip32().concat_bits(
			[(self.read_mem_8(addr + self.byte_significance[x] -1), 8) 
				for x in range(4)
			]
		)[0]

	def read_mem_16(self, addr: int) -> uint16:

		bm = BitManip32()

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
	bm = BitManip32()
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
