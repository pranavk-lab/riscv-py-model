import isa
from bit_manipulation import BitManip, XLen
import numpy as np
from rv64ui import NOP

class RV64SI:

	def __init__(self):
		self.name = "RISC-V 64-bit user"
		self.uint = np.uint64
		self.int = np.int64
		self.NOP = NOP
		self.instructions = {0x1C : System}

	def get_instructions(self):
		return self.instructions


#TODO: implement system calls
class System(isa.InstructionTemplate):

	def __init__(self, instr, core_state):
		super().__init__(instr, core_state)
	
	def execute(self):
		return super().execute()

	def dump_instr(self):
		return super().dump_instr()
