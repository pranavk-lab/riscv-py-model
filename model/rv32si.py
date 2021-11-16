import isa
from bit_manipulation import BitManip, XLen
import numpy as np
from rv64si import RV64SI

class RV32SI:

	def __init__(self):
		self.name = "RISC-V 64-bit user"
		self.uint = np.uint32
		self.int = np.int32
		self.instructions = RV64SI().get_instructions()

	def get_instructions(self):
		return self.instructions