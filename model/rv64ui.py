import isa
from bit_manipulation import BitManip, XLen
import numpy as np

class RV64UI:

	def __init__(self):
		self.name = "RISC-V 64-bit user"
		self.uint = np.uint64
		self.int = np.int64
		self.NOP = NOP_64
		self.instructions = {
				0x18 : ConditionalBranch_64,
				0x1b : JumpAndLink_64,
				0x19 : JumpAndLinkRegsiter_64,
				0x0d : LoadUpperImm_64,
				0x05 : AddUpperImmPC_64,
				0x04 : RegImmInt_64,
				0x0C : RegRegInt_64,
				0x00 : Load_64,
				0x08 : Store_64,
				0x03 : Fence_64,
				0x01 : NOP_64
			}

	def get_instructions(self):
		return self.instructions


class ConditionalBranch_64(isa.InstructionTemplate):

	def beq(self):
		if np.int64(self.src1_val) == np.int64(self.src2_val):
			self.core_state.incr_PC(self.offset)
		else:
			self.core_state.incr_PC()
	
	def bne(self):
		if np.int64(self.src1_val) != np.int64(self.src2_val):
			self.core_state.incr_PC(self.offset)
		else:
			self.core_state.incr_PC()
		
	def blt(self):
		if np.int64(self.src1_val) < np.int64(self.src2_val):
			self.core_state.incr_PC(self.offset)
		else:
			self.core_state.incr_PC()
	
	def bge(self):
		if np.int64(self.src1_val) > np.int64(self.src2_val):
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

	def __init__(self, instr: np.uint64, core_state):

		bm = BitManip(XLen._64BIT)
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
	

class JumpAndLinkRegsiter_64(isa.InstructionTemplate):

	def __init__(self, instr: np.uint64, core_state):

		bm = BitManip(XLen._64BIT)
		self.core_state = core_state

		self.offset = bm.sign_extend_nbit_2_int(
			bm.get_sub_bits_from_instr(instr, 31, 20)
		)

		self.src, w5 = bm.get_sub_bits_from_instr(instr, 19, 15)

		self.dst, w5 = bm.get_sub_bits_from_instr(instr, 11, 7)

		self.src1_val = core_state.REG_FILE[self.src]

		self.PC_state = self.core_state.PC

	def execute(self):
		self.core_state.REG_FILE[self.dst] = self.core_state.PC + 4

		# Add offset and content in REG_FILE[src]
		result = self.offset + np.int64(self.core_state.REG_FILE[self.src])

		# Set the least significant bit of result to 0. 
		# Don't ask me why? It's in the RISCV specification. 
		self.core_state.PC = np.bitwise_and(result, 0xfffffffe)
	
	def dump_instr(self) -> tuple:
		return (
			("instr", "JALR instruction"),
			("offset", self.offset),
			("src", self.src),
			("src_val", self.src1_val),
			("dest", self.dst)
		)


class JumpAndLink_64(isa.InstructionTemplate):

	def __init__(self, instr: np.uint64, core_state):
		bm = BitManip(XLen._64BIT)

		self.offset = bm.sign_extend_nbit_2_int(bm.concat_bits([
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

class LoadUpperImm_64(isa.InstructionTemplate):

	def __init__(self, instr: np.uint64, core_state):
		bm = BitManip(XLen._64BIT)

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

class AddUpperImmPC_64(isa.InstructionTemplate): 

	def __init__(self, instr: np.uint64, core_state):
		bm = BitManip(XLen._64BIT)

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


class RegImmInt_64(isa.InstructionTemplate):

	def addi(self) -> np.uint64:
		return np.int64(self.src_val) + self.signed_imm_arith
	
	def slli(self) -> np.uint64:
		return np.left_shift(self.src_val, np.uint64(self.shamt))

	def slti(self) -> np.uint64:
		if np.int64(self.src_val) < self.signed_imm_arith:
			return 1
		else:
			return 0

	def sltiu(self) -> np.uint64:
		if self.src_val < self.unsigned_imm_arith:
			return 1
		else:
			return 0
		
	def xori(self) -> np.uint64:
		return np.bitwise_xor(self.src_val, self.unsigned_imm_arith)

	def srai_srli(self) -> np.uint64:
		# SRA
		if self.imm_shift_v == 32:
			return np.right_shift(np.int64(self.src_val), np.int64(self.shamt))
		
		# SRL
		elif self.imm_shift_v == 0:
			return np.right_shift(
				np.uint64(self.src_val), np.uint64(self.shamt)
			)
		
		else:
			raise ValueError(f" immediate[11:5] needs to be either 32 or 0. \
				Actual imm = {self.imm_shift_v}")
	
	def ori(self) -> np.uint64:
		return np.bitwise_or(self.src_val, self.unsigned_imm_arith)
		
	def andi(self) -> np.uint64:
		return np.bitwise_and(self.src_val, self.unsigned_imm_arith)

	def __init__(self, instr: np.uint64, core_state):
		bm = BitManip(XLen._64BIT)

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
		self.unsigned_imm_arith = bm.sign_extend_nbit_2_unsigned_int(self.imm_arith_v)
		self.signed_imm_arith = bm.sign_extend_nbit_2_int(self.imm_arith_v)
		self.shamt, w5 = bm.get_sub_bits_from_instr(instr, 24, 20)
		self.src, w5 = bm.get_sub_bits_from_instr(instr, 19, 15)
		self.dest, w5 = bm.get_sub_bits_from_instr(instr, 11, 7)
		self.src_val = np.uint64(core_state.REG_FILE[self.src])
		self.core_state = core_state

	def execute(self):
		# Execute instruction based on funct3
		self.core_state.REG_FILE[self.dest] = (
			np.uint64(self.int_function[self.funct3]())
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

class RegRegInt_64(isa.InstructionTemplate):

	def add_sub(self) -> np.uint64:

		# ADD
		if self.funct7 == 0:
			return np.int64(self.src1_val) + np.int64(self.src2_val)

		# SUB
		elif self.funct7 == 32:
			return np.int64(self.src1_val) - np.int64(self.src2_val)

		else:
			raise ValueError(f"funct7 needs to be either 32 or 0. \
				Actual funct7 = {self.funct7}")
		
	def sll(self) -> np.uint64:
		return np.left_shift(self.src1_val, np.uint64(self.shift_val))
	
	def slt(self) -> np.uint64:
		if np.int64(self.src1_val) < np.int64(self.src2_val):
			return 1
		else:
			return 0

	def sltu(self) -> np.uint64:
		if self.src1_val < self.src2_val:
			return 1
		else:
			return 0
	
	def xor_(self) -> np.uint64:
		return np.bitwise_xor(self.src1_val, self.src2_val)

	def sra_srl(self) -> np.uint64:

		if self.funct7 == 32:
			return np.right_shift(np.int64(self.src1_val), np.int64(self.shift_val))
		
		elif self.funct7 == 0:
			return np.right_shift(self.src1_val, np.uint64(self.shift_val))
		
		else:
			raise ValueError(f" funct7 needs to be either 32 or 0. \
			Actual funct7 = {self.funct7}")
	
	def or_(self) -> np.uint64:
		return np.bitwise_or(self.src1_val, self.src2_val)
	
	def and_(self) -> np.uint64:
		return np.bitwise_and(self.src1_val, self.src2_val)

	def __init__(self, instr: np.uint64, core_state):
		bm = BitManip(XLen._64BIT)

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

		self.src1_val = np.uint64(core_state.REG_FILE[self.src1])

		self.src2_val = np.uint64(core_state.REG_FILE[self.src2])

		self.shift_val, w5 = bm.get_sub_bits_from_instr(self.src2_val, 4, 0)

		self.core_state = core_state

	def execute(self):
		self.core_state.REG_FILE[self.dest] = (
			np.uint64(self.int_function[self.funct3]())
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

class Load_64(isa.InstructionTemplate):

	def lb(self) -> np.uint64:
		return self.bm.sign_extend_nbit_2_unsigned_int((self.lbu(), 8))

	def lh(self) -> np.uint64:
		return self.bm.sign_extend_nbit_2_unsigned_int((self.lhu(), 16))
	
	def lw(self) -> np.uint64:
		return self.core_state.memory.read_mem_64(self.mem_addr)
	
	def lhu(self) -> np.uint64:
		return self.core_state.memory.read_mem_16(self.mem_addr)
	
	def lbu(self) -> np.uint64:
		return self.core_state.memory.read_mem_8(self.mem_addr)

	def __init__(self, instr: np.uint64, core_state):

		self.bm = BitManip(XLen._64BIT)

		self.core_state = core_state

		imm, w12 = self.bm.get_sub_bits_from_instr(instr, 31, 20)
		self.src, w5 = self.bm.get_sub_bits_from_instr(instr, 19, 15)
		self.dest, w5 = self.bm.get_sub_bits_from_instr(instr, 11, 7)
		self.funct3, w3 = self.bm.get_sub_bits_from_instr(instr, 14, 12)
		self.mem_addr = (
			core_state.REG_FILE[self.src] + 
			self.bm.sign_extend_nbit_2_int((imm, w12))
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
			np.uint64(self.load_function[self.funct3]())
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


class Store_64(isa.InstructionTemplate):

	def sb(self):
		self.core_state.memory.write_mem_8(self.mem_addr, np.uint8(self.src2_val))

	def sh(self):
		self.core_state.memory.write_mem_16(self.mem_addr, np.uint16(self.src2_val))

	def sw(self):
		self.core_state.memory.write_mem_64(self.mem_addr, np.uint64(self.src2_val))

	def __init__(self, instr: np.uint64, core_state):
		bm = BitManip(XLen._64BIT)

		imm = bm.sign_extend_nbit_2_int(bm.concat_bits([
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


class NOP_64(isa.InstructionTemplate):

	def __init__(self, instr: np.uint64, core_state):
		self.PC_state = core_state.PC
	
	def execute(self):
		return super().execute()
	
	def dump_instr(self) -> tuple:
		return super().dump_instr()

class Fence_64(isa.InstructionTemplate):
	def __init__(instr: np.uint64, core_state):
		bm = BitManip(XLen._64BIT)

		#TODO: finish this routine
		return core_state	

	def execute(self):
		pass

