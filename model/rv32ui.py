import isa
from bit_manipulation import BitManip, XLen
import numpy as np
from rv64ui import RV64UI

class RV32UI:

	def __init__(self):
		self.name = "RISC-V 32 bit user "
		self.uint = np.uint32
		self.int = np.int32
		rv64ui = RV64UI()
		self.NOP = rv64ui.NOP
		self.instructions = rv64ui.base_instructions

	def get_instructions(self):
		return self.instructions
