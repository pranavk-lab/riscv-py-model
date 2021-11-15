#!/usr/bin/env python3
from context import cpu
#!/usr/bin/env python3
from context import BitManip, XLen
import unittest

from cpu import Endianess


class TestMemory(unittest.TestCase):

    VAL_32 = 0xAA5533FF
    VAL_64 = 0xAA5533FF889944EE

    def setUp(self) -> None:
        self.little_mem_32 = cpu.Memory(endianess=Endianess.LITTLE, xlen=XLen._32BIT)
        self.little_mem_64 = cpu.Memory(endianess=Endianess.LITTLE, xlen=XLen._64BIT)

        self.big_mem_32 = cpu.Memory(endianess=Endianess.BIG, xlen=XLen._32BIT)
        self.big_mem_64 = cpu.Memory(endianess=Endianess.BIG, xlen=XLen._64BIT)

    def test_write_mem_word_little_32_bit(self):
        mem = self.little_mem_32
        bytes_per_word = mem.xlen.value//8
        self._test_write_mem_word(mem, self.VAL_64, bytes_per_word)

    def test_write_mem_half_word_little_32_bit(self):
        mem = self.little_mem_32
        bytes_per_word = mem.xlen.value//8
        self._test_write_mem_word(mem, self.VAL_64, bytes_per_word//2)

    def test_write_mem_byte_little_32_bit(self):
        mem = self.little_mem_32
        bytes_per_word = mem.xlen.value//8
        self._test_write_mem_word(mem, self.VAL_64, bytes_per_word//4)

    def test_write_mem_word_little_64_bit(self):
        mem = self.little_mem_64
        bytes_per_word = mem.xlen.value//8
        self._test_write_mem_word(mem, self.VAL_64, bytes_per_word)
    
    def test_write_mem_half_word_little_64_bit(self):
        mem = self.little_mem_64
        bytes_per_word = mem.xlen.value//8
        self._test_write_mem_word(mem, self.VAL_64, bytes_per_word//2)

    def test_write_mem_quart_word_little_64_bit(self):
        mem = self.little_mem_64
        bytes_per_word = mem.xlen.value//8
        self._test_write_mem_word(mem, self.VAL_64, bytes_per_word//4)

    def test_write_mem_byte_little_64_bit(self):
        mem = self.little_mem_64
        bytes_per_word = mem.xlen.value//8
        self._test_write_mem_word(mem, self.VAL_64, bytes_per_word//8)

    def test_write_mem_word_big_32_bit(self):
        mem = self.big_mem_32
        bytes_per_word = mem.xlen.value//8
        self._test_write_mem_word(mem, self.VAL_32, bytes_per_word)

    def test_write_mem_half_word_big_32_bit(self):
        mem = self.big_mem_32
        bytes_per_word = mem.xlen.value//8
        self._test_write_mem_word(mem, self.VAL_32, bytes_per_word//2)

    def test_write_mem_byte_big_32_bit(self):
        mem = self.big_mem_32
        bytes_per_word = mem.xlen.value//8
        self._test_write_mem_word(mem, self.VAL_32, bytes_per_word//4)

    def test_write_mem_word_big_64_bit(self):
        mem = self.big_mem_64
        bytes_per_word = mem.xlen.value//8
        self._test_write_mem_word(mem, self.VAL_64, bytes_per_word)

    def test_write_mem_half_word_big_64_bit(self):
        mem = self.big_mem_64
        bytes_per_word = mem.xlen.value//8
        self._test_write_mem_word(mem, self.VAL_64, bytes_per_word//2)

    def test_write_mem_quart_word_big_64_bit(self):
        mem = self.big_mem_64
        bytes_per_word = mem.xlen.value//8
        self._test_write_mem_word(mem, self.VAL_64, bytes_per_word//4)

    def test_write_mem_byte_big_64_bit(self):
        mem = self.big_mem_64
        bytes_per_word = mem.xlen.value//8
        self._test_write_mem_word(mem, self.VAL_64, bytes_per_word//8)

    def _test_write_mem_word(self, mem: cpu.Memory, val, word_len):
        addr = mem.mem_size//2
        mem.write_mem_word(addr, val, word_len)
        val_read = mem.read_mem_word(addr, word_len)
        self.assertEqual(
            val_read, mem.bm.get_sub_bits_from_instr(val, 8*word_len-1, 0)[0]
        )


unittest.main()