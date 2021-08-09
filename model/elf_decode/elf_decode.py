#!/usr/bin/env python3
import glob
import os
import sys
from elftools.elf.elffile import ELFFile
from elftools.elf.segments import Segment

class ELF_DECODE:
	
	def __init__(self):
		self.test_files = []
	
	def get_elf_files(self, user_path):
		test_files = []
		for x in glob.glob(os.path.join(user_path, 'rv32ui-*')):
			if not x.endswith('dump'):
				self.test_files.append(x)
	
	def get_test_cases(self, test_file):
		with open(test_file, 'rb') as elffile:
			elf = ELFFile(elffile)
			symtab = elf.get_section_by_name('.symtab')
			if not symtab:
				print('No .symtab found in elf')
				sys.exit(1)
		
			test_name = []
			test_value = []
			for sym in symtab.iter_symbols():
				if 'test' in sym.name:
					test_name.append(sym.name)
					test_value.append(sym['st_value'])
					
				
			if len(test_name) == 0:
				print(' No test cases found in elf')
				sys.exit(1)
					
			file_offset = []
			for tests in test_value:
				for seg in elf.iter_segments():
					if seg.header['p_type'] != 'PT_LOAD':
						continue
				
					if tests >= seg['p_vaddr'] and tests < seg['p_vaddr'] + seg['p_filesz']:
						file_offset.append(tests - seg['p_vaddr'] + seg['p_offset'])
						break
		
			if len(file_offset) != len(test_value):
				print('Not all offsets found')
			
			test_table = [] 
			for x in range(len(file_offset)):
				test_table.append((test_name[x], test_value[x], file_offset[x]))

			return test_table
	
	def get_word_list(self, test_path):
		with open(test_path, 'rb') as test:
			test_bytes = test.read()
			return [test_bytes[i:i+4].hex() for i in range(0, len(test_bytes), 4)]

	def get_assm_code(self, test_cases, elf_file):
		elf_words = self.get_word_list(elf_file)
		with open('elf.words', 'w') as elf_debug:
			for x in elf_words:
				elf_debug.writelines(f'{x}\n')
		# List of code per test case, which is a list. 
		test_code = []
		for x in range(len(test_cases)-1):
			test_code.append(elf_words[int(test_cases[x][2]/4):int(test_cases[x+1][2]/4)])
		return test_code

	def get_bytes_from_word(self, word):
		return [word[i:i+2] for i in range(0, len(word), 2)]

	def reverse_string(self, word):
		stack = []
		for i in range(0, len(word), 2):
			stack.append(word[i:i+2])
		
		word = ''
		for i in range(len(stack)):
			word += stack.pop()
		return word
		
	def big2little(self, assm_code):
		for x in range(len(assm_code)):
			assm_code[x] = self.reverse_string(assm_code[x])
		return assm_code
			
			
if __name__ == '__main__':
	dict_ = []
	assm_code = []
	x = ELF_DECODE()
	riscv_tests = '/home/prankov/work/riscv/riscv-tests/isa'
	x.get_elf_files(riscv_tests)
	for y in x.test_files:
		if y == os.path.join(riscv_tests, 'rv32ui-p-add'):
			dict_ = x.get_test_cases(y)
			assm_code = x.get_assm_code(dict_, y)
			litte_end = x.big2little(assm_code[0])
	print(dict_)
	print('+++++++++++++++++++++++++++++++')
	print(assm_code)
	print('+++++++++++++++++++++++++++++++')
	print(litte_end)

	
				
		
		



		


