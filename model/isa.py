from abc import ABC, abstractmethod
from enum import Enum

from enum import Enum

class InstructionType(Enum):
	BASE_INSTR = 0
	DERIVED_INSTR = 1


class InstructionTemplate(ABC):
	""" Template for instrutions """

	@abstractmethod
	def __init__(self, instr, core_state):
		pass

	@abstractmethod
	def execute(self):
		pass

	@abstractmethod
	def dump_instr(self):
		return (())