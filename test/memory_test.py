#!/usr/bin/env python3
from context import cpu
#!/usr/bin/env python3
from context import BitManip, XLen
import unittest

from cpu import Endianess


class TestMemory(unittest.TestCase):

    VAL_32 = 0xAA5533FF
    VAL_64 = 0xAA5533FF889944EE

    def test_write_mem_word_little_32_bit(self):
        self._test_write_mem_word(cpu.Memory(xlen=XLen._32BIT), self.VAL_32)

    def test_write_mem_word_little_64_bit(self):
        self._test_write_mem_word(cpu.Memory(), self.VAL_64)

    def test_write_mem_word_big_32_bit(self):
        self._test_write_mem_word(
            cpu.Memory(endianess=Endianess.BIG, xlen=XLen._32BIT), 
            self.VAL_32
        )

    def test_write_mem_word_big_64_bit(self):
        self._test_write_mem_word(
            cpu.Memory(endianess=Endianess.BIG),
            self.VAL_64
        )

    def _test_write_mem_word(self, mem: cpu.Memory, val):
        mem.write_mem_word(mem.mem_size//2, val, mem.xlen.value//8)
        val_read = mem.read_mem_word(mem.mem_size//2, mem.xlen.value//8)
        self.assertEqual(val_read, val)


unittest.main()