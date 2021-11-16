from numpy.core.fromnumeric import resize
import isa
from bit_manipulation import BitManip, XLen
import numpy as np

class RV64UI:

	def __init__(self):
		self.name = "RISC-V 64-bit user"
		self.uint = np.uint64
		self.int = np.int64
		self.NOP = NOP
		self.base_instructions = {
				0x18 : ConditionalBranch,
				0x1b : JumpAndLink,
				0x19 : JumpAndLinkRegsiter,
				0x0d : LoadUpperImm,
				0x05 : AddUpperImmPC,
				0x04 : RegImmInt,
				0x0C : RegRegInt,
				0x00 : Load,
				0x08 : Store,
				0x03 : Fence,
				0x01 : NOP
			}

		self.compat_32_instructions = {
				0x06 : RegImmInt32Compat,
				0x0E : RegRegInt32Compat,
			}

	def get_instructions(self):
		return {**self.base_instructions, **self.compat_32_instructions}


class ConditionalBranch(isa.InstructionTemplate):

	def beq(self):
		if self.src1_val == self.src2_val:
			self.core.incr_PC(self.bit_fields.imm)
		else:
			self.core.incr_PC()
	
	def bne(self):
		if (self.src1_val) != (self.src2_val):
			self.core.incr_PC(self.bit_fields.imm)
		else:
			self.core.incr_PC()
		
	def blt(self):
		if (self.src1_val) < (self.src2_val):
			self.core.incr_PC(self.bit_fields.imm)
		else:
			self.core.incr_PC()
	
	def bge(self):
		if (self.src1_val) > (self.src2_val):
			self.core.incr_PC(self.bit_fields.imm)
		else:
			self.core.incr_PC()

	def bltu(self):
		if self.src1_uval < self.src2_uval:
			self.core.incr_PC(self.bit_fields.imm)
		else:
			self.core.incr_PC()
	
	def bgeu(self):
		if self.src1_uval > self.src2_uval:
			self.core.incr_PC(self.bit_fields.imm)
		else:
			self.core.incr_PC()

	def __init__(self, instr: np.uint32, core_state):
		bm = BitManip(core_state.xlen)
		self.branch_function = {
			0 : self.beq,
			1 : self.bne,
			4 : self.blt,
			5 : self.bge,
			6 : self.bltu,
			7 : self.bgeu 
		}
		self.core = core_state
		self.PC_state = self.core.PC
		self.bit_fields = isa.BTypeInstr(bm, instr)
		self.src1_uval = core_state.REG_FILE[self.bit_fields.src1]
		self.src1_val = self.core.isa.int(self.src1_uval)

		# self.src2 = bm.get_sub_bits_from_instr(instr, 24, 20)[0] 
		self.src2_uval = core_state.REG_FILE[self.bit_fields.src2]
		self.src2_val = self.core.isa.int(self.src2_uval)
	
	def execute(self):

		if self.bit_fields.funct3 not in self.branch_function.keys():
			raise ValueError(f" Invalid funct3 = {self.bit_fields.funct3}")

		self.branch_function[self.bit_fields.funct3]()
	
	def dump_instr(self) -> tuple:
		return (
			("instr", "branch instruction"),
			("funct3", self.bit_fields.funct3),
			("offset", self.bit_fields.imm),
			("src1", self.bit_fields.src1),
			("src1_val", self.src1_val),
			("src2", self.bit_fields.src2),
			("src2_val", self.src2_val)
		)
	

class JumpAndLinkRegsiter(isa.InstructionTemplate):

	def __init__(self, instr: np.uint32, core_state):
		self.bm = BitManip(core_state.xlen)
		self.core = core_state
		self.bit_fields = isa.ITypeInstr(self.bm, instr)
		self.src1_val = core_state.REG_FILE[self.bit_fields.src]
		self.PC_state = self.core.PC

	def execute(self):
		self.core.REG_FILE[self.bit_fields.dest] = self.core.PC + 4

		# Add offset and content in REG_FILE[src]
		result = (
			self.bit_fields.imm + 
			self.core.isa.int(self.core.REG_FILE[self.bit_fields.src])
		)

		# Set the least significant bit of result to 0. 
		# Don't ask me why? It's in the RISCV specification. 
		self.core.PC = np.bitwise_and(
			self.core.isa.uint(result), 
			np.left_shift(self.bm.uint_max, self.core.isa.uint(1))
		)
	
	def dump_instr(self) -> tuple:
		return (
			("instr", "JALR instruction"),
			("offset", self.bit_fields.imm),
			("src", self.bit_fields.src),
			("src_val", self.src1_val),
			("dest", self.bit_fields.dest)
		)


class JumpAndLink(isa.InstructionTemplate):

	def __init__(self, instr: np.uint32, core_state):
		bm = BitManip(core_state.xlen)
		self.bit_fields = isa.JTypeInstr(bm, instr)
		self.core = core_state

	def execute(self):
		self.core.REG_FILE[self.bit_fields.dst] = self.core.PC + 4
		self.core.incr_PC(self.bit_fields.imm)

	def dump_instr(self) -> tuple:
		return (
			("instr", "JAL instruction"),
			("offset", self.bit_fields.imm),
			("dest", self.bit_fields.dst)
		)

class LoadUpperImm(isa.InstructionTemplate):

	def __init__(self, instr: np.uint32, core_state):
		bm = BitManip(core_state.xlen)
		self.core = core_state
		self.bit_fields = isa.UTypeInstr(bm, instr)

	def execute(self):
		self.core.REG_FILE[self.bit_fields.dst] = self.bit_fields.u_imm
		self.core.incr_PC()
	
	def dump_instr(self) -> tuple:
		return (
			("instr", "LUI instruction"),
			("imm", self.bit_fields.u_imm),
			("dst", self.bit_fields.dst),
		)

class AddUpperImmPC(isa.InstructionTemplate): 

	def __init__(self, instr: np.uint32, core_state):
		bm = BitManip(core_state.xlen)
		self.core = core_state
		self.bit_fields = isa.UTypeInstr(bm, instr)

	def execute(self):
		self.core.REG_FILE[self.bit_fields.dst] = self.core.isa.uint(
			self.bit_fields.u_imm + self.core.incr_PC()
		)

	def dump_instr(self) -> tuple:
		return (
			("instr", "AUI instruction"),
			("imm", self.bit_fields.u_imm),
			("dst", self.bit_fields.dst),
		)


class RegImmInt(isa.InstructionTemplate):

	def addi(self) -> int:
		return self.bm.int(self.src_val) + self.signed_imm_arith
	
	def slli(self) -> int:
		return np.left_shift(self.src_val, self.bit_fields.shamt)

	def slti(self) -> int:
		if self.bm.int(self.src_val) < self.signed_imm_arith:
			return 1
		else:
			return 0

	def sltiu(self) -> int:
		return int(self.src_val < self.unsigned_imm_arith)
		
	def xori(self) -> int:
		return np.bitwise_xor(self.src_val, self.unsigned_imm_arith)

	def srai_srli(self) -> int:
		# SRA
		if self.bit_fields.shift_imm == 32:
			return np.right_shift(
				self.bm.int(self.src_val), self.bm.int(self.bit_fields.shamt)
			)
		
		# SRL
		elif self.bit_fields.shift_imm == 0:
			return np.right_shift(self.src_val, self.bit_fields.shamt)
		
		else:
			raise ValueError(f" immediate[11:5] needs to be either 32 or 0. \
				Actual imm = {self.bit_fields.shift_imm}")
	
	def ori(self) -> int:
		return np.bitwise_or(self.src_val, self.unsigned_imm_arith)
		
	def andi(self) -> int:
		return np.bitwise_and(self.src_val, self.unsigned_imm_arith)

	def __init__(
		self, 
		instr: np.uint32, 
		core_state, 
		base_instr = isa.InstructionType.BASE_INSTR):

		# If rv64ui 32-bit compatible instruction
		self.bm = BitManip(XLen._32BIT)

		if base_instr == isa.InstructionType.BASE_INSTR:
			self.bm = BitManip(core_state.xlen)

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
		self.core = core_state
		self.bit_fields = isa.ITypeInstr(self.bm, instr)
		self.unsigned_imm_arith = self.bm.sign_extend_nbit_2_unsigned_int(
			(self.bit_fields.imm, self.bit_fields.imm_width)
		)
		self.signed_imm_arith = self.bm.sign_extend_nbit_2_int(
			(self.bit_fields.imm, self.bit_fields.imm_width)
		)
		self.src_val = self.bm.uint(core_state.REG_FILE[self.bit_fields.src])
		self.base_instr = base_instr

	def execute(self):
		"""Execute instruction based on funct3"""

		self.core.REG_FILE[self.bit_fields.dest] = self.bm.uint(
			self.int_function[self.bit_fields.funct3]()
		)

		self.core.incr_PC()
	
	def dump_instr(self) -> tuple:
		if self.funct3 == 1 or self.funct3 == 5:
			return (
				("instr", "shift instruction"),
				("funct3", self.bit_fields.funct3),
				("src1", self.bit_fields.src),
				("src1_val", self.src_val),
				("dest", self.bit_fields.dest),
				("shift", self.bit_fields.shamt),
				("imm", self.bit_fields.shift_imm)
			)

		return (
			("instr", "ADDI/SLT/LOGICAL instruction"),
			("funct3", self.bit_fields.funct3),
			("src1", self.bit_fields.src),
			("src1_val", self.src_val),
			("dest", self.bit_fields.dest),
			("imm", self.bit_fields.imm)
		)

class RegRegInt(isa.InstructionTemplate):

	def add_sub(self) -> int:

		# ADD
		if self.bit_fields.funct7 == 0:
			return (self.bm.int(self.src1_val) 
				+ self.bm.int(self.src2_val)
			)

		# SUB
		elif self.bit_fields.funct7 == 32:
			return (self.bm.int(self.src1_val) 
				- self.bm.int(self.src2_val)
			)

		else:
			raise ValueError(f"funct7 needs to be either 32 or 0. \
				Actual funct7 = {self.bit_fields.funct7}")
		
	def sll(self) -> int:
		return np.left_shift(self.src1_val, self.shift_val)
	
	def slt(self) -> int:
		return int(self.bm.int(self.src1_val) < self.bm.int(self.src2_val))

	def sltu(self) -> int:
		return int(self.src1_val < self.src2_val)
	
	def xor_(self) -> int:
		return np.bitwise_xor(self.src1_val, self.src2_val)

	def sra_srl(self) -> int:

		if self.bit_fields.funct7 == 32:
			return np.right_shift(
				self.bm.int(self.src1_val), 
				self.bm.int(self.shift_val)
			)
		
		elif self.bit_fields.funct7 == 0:
			return np.right_shift(self.src1_val, self.shift_val)
		
		else:
			raise ValueError(f" funct7 needs to be either 32 or 0. \
			Actual funct7 = {self.bit_fields.funct7}")
	
	def or_(self) -> int:
		return np.bitwise_or(self.src1_val, self.src2_val)
	
	def and_(self) -> int:
		return np.bitwise_and(self.src1_val, self.src2_val)

	def __init__(
		self, 
		instr: np.uint32, 
		core_state, 
		base_instr = isa.InstructionType.BASE_INSTR):

		# If rv64ui 32-bit compatible instruction
		self.bm = BitManip(XLen._32BIT)

		if base_instr == isa.InstructionType.BASE_INSTR:
			self.bm = BitManip(core_state.xlen)

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
		self.core = core_state
		self.bit_fields = isa.RTypeInstr(self.bm, instr)
		self.src1_val = self.bm.uint(core_state.REG_FILE[self.bit_fields.src1])
		self.src2_val = self.bm.uint(core_state.REG_FILE[self.bit_fields.src2])
		self.shift_val, w5 = self.bm.get_sub_bits_from_instr(
			self.src2_val, 4, 0
		)


	def execute(self):
		self.core.REG_FILE[self.bit_fields.dest] = self.bm.uint(
			self.int_function[self.bit_fields.funct3]()
		)
		self.core.incr_PC()
	
	def dump_instr(self) -> tuple:
		return (
			("instr", "Reg Reg int instruction"),
			("funct3", self.bit_fields.funct3),
			("funct7", self.bit_fields.funct7),
			("src1", self.bit_fields.src1),
			("src1_val", self.src1_val),
			("src2", self.bit_fields.src2),
			("src2_val", self.src2_val),
			("dest", self.bit_fields.dest),
		)

class Load(isa.InstructionTemplate):

	def lb(self) -> int:
		return self.bm.sign_extend_nbit_2_unsigned_int((self.lbu(), 8))

	def lh(self) -> int:
		return self.bm.sign_extend_nbit_2_unsigned_int((self.lhu(), 16))
	
	def lw(self) -> int:
		return self.bm.sign_extend_nbit_2_unsigned_int(
			(self.core.memory.read_mem_32(self.mem_addr), self.core.xlen.value)
		)
	
	def ld(self) -> int:
		if self.core.xlen != XLen._64BIT:
			raise RuntimeError("Cannot run ld, this is a 32 bit cpu")

		return self.core.memory.read_mem_64(self.mem_addr)

	def lwu(self) -> int:
		if self.core.xlen != XLen._64BIT:
			raise RuntimeError("Cannot run lwu, this is a 32 bit cpu")

		return self.core.memory.write_mem_32(self.mem_addr)
	
	def lhu(self) -> int:
		return self.core.memory.read_mem_16(self.mem_addr)
	
	def lbu(self) -> int:
		return self.core.memory.read_mem_8(self.mem_addr)

	def __init__(self, instr: np.uint32, core_state):

		self.bm = BitManip(core_state.xlen)
		self.load_function = {
			0 : self.lb,
			1 : self.lh,
			3 : self.ld,
			2 : self.lw,
			4 : self.lbu,
			5 : self.lhu, 
			6 : self.lwu
		}
		self.core = core_state

		self.bit_fields = isa.ITypeInstr(self.bm, instr)

		self.mem_addr = (
			core_state.REG_FILE[self.bit_fields.src] + 
			self.bm.sign_extend_nbit_2_int(
				(self.bit_fields.imm, self.bit_fields.imm_width)
			)
		)

	def execute(self):
		if self.bit_fields.funct3 not in self.load_function.keys():
			raise ValueError(f" Invalid funct3. \
				Must be {self.load_function.keys()}.\
					Actual funct3 = {self.bit_fields.funct3}")

		self.core.REG_FILE[self.bit_fields.dest] = (
			self.core.isa.uint(self.load_function[self.bit_fields.funct3]())
		)
		self.core.incr_PC()
	
	def dump_instr(self) -> tuple:
		return (
			("instr", "load instruction"),
			("funct3", self.bit_fields.funct3),
			("src1", self.bit_fields.src),
			("mem_addr", self.mem_addr),
			("dest", self.bit_fields.dest),
		)


class Store(isa.InstructionTemplate):

	def sb(self):
		self.core.memory.write_mem_8(self.mem_addr, np.uint8(self.src2_val))

	def sh(self):
		self.core.memory.write_mem_16(self.mem_addr, np.uint16(self.src2_val))

	def sw(self):
		self.core.memory.write_mem_32(self.mem_addr, np.uint32(self.src2_val))

	def sd(self):
		if self.core.xlen != XLen._64BIT:
			raise RuntimeError("Cannot run sd, this is a 32 bit cpu")

		self.core.memory.write_mem_64(self.mem_addr, np.uint64(self.src2_val))

	def __init__(self, instr: np.uint32, core_state):
		self.bm = BitManip(core_state.xlen)
		self.core = core_state
		self.bit_fields = isa.STypeInstr(self.bm, instr)
		self.src1_val = core_state.REG_FILE[self.bit_fields.src1]
		self.src2_val = core_state.REG_FILE[self.bit_fields.src2]
		self.mem_addr = self.bit_fields.imm + self.src1_val
		self.store_function = {
			0 : self.sb,
			1 : self.sh,
			2 : self.sw,
			3 : self.sd
		}

	def execute(self):
				
		if self.bit_fields.funct3 not in self.store_function.keys():
			raise ValueError(f"funct3 is out of scope. \
				Must be 0 <= funct3 <= 2. Actual funct3 = {self.bit_fields.funct3}")

		# Run store instruction
		self.store_function[self.bit_fields.funct3]()
		
		self.core.incr_PC()
	
	def dump_instr(self) -> tuple:
		return (
			("instr", "store instruction"),
			("funct3", self.bit_fields.funct3),
			("src1", self.bit_fields.src1),
			("src1_val", self.src1_val),
			("src2", self.bit_fields.src2),
			("src2_val", self.src2_val),
			("mem_addr", self.mem_addr),
		)


class NOP(isa.InstructionTemplate):

	def __init__(self, instr: np.uint32, core_state):
		self.PC_state = core_state.PC
	
	def execute(self):
		return super().execute()
	
	def dump_instr(self) -> tuple:
		return super().dump_instr()


class Fence(isa.InstructionTemplate):
	def __init__(instr: np.uint32, core_state):
		bm = BitManip(core_state.xlen)

		#TODO: finish this routine
		return core_state	

	def execute(self):
		pass


class RegImmInt32Compat(RegImmInt):

	def __init__(self, instr: np.uint64, core_state):
		super().__init__(instr, core_state, isa.InstructionType.DERIVED_INSTR)
		
	def execute(self):
		super().execute()

	def dump_instr(self) -> tuple:
		return super().dump_instr()


class RegRegInt32Compat(RegRegInt):

	def __init__(self, instr: np.uint64, core_state):
		super().__init__(instr, core_state, isa.InstructionType.DERIVED_INSTR)

	def execute(self):
		super().execute()
	
	def dump_instr(self) -> tuple:
		return super().dump_instr()