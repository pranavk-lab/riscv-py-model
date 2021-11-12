from abc import ABC, abstractmethod

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