import struct
from typing import List
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.triaxis.lexicon import GLLL

def glll_to_int(glll_code: str) -> int:
    """Convert GLLL code like 'H64:r32' to integer 0x32."""
    hex_part = glll_code.split(':')[1]  # 'r32'
    return int(hex_part[1:], 16)  # '32' -> 0x32

# Hadamard Opcode Mappings - Aligned with Triaxis GLLL (Praxica-H)
HADAMARD_OPCODES = {
    # Existing mappings aligned to GLLL where possible
    "∂": 0x01,  # Keep for now, custom
    "∇": 0x02,
    "∇²": 0x03,
    "φ": 0x04,
    "↻": 0x05,
    "⊕": 0x06,
    "⊖": 0x07,
    "◯": 0x08,
    "∆": 0x09,
    "□": 0x0A,
    "⇒": 0x0B,
    "*": glll_to_int(GLLL.MUL),  # 0x34
    "/": glll_to_int(GLLL.DIV),  # 0x35
    "+": glll_to_int(GLLL.ADD),  # 0x32
    "-": glll_to_int(GLLL.SUB),  # 0x33
    "=": glll_to_int(GLLL.STORE),  # 0x17
    "(": glll_to_int(GLLL.PUSH),  # 0x22
    ")": glll_to_int(GLLL.POP),   # 0x23
    "ricci": 0x20,
    "lie": 0x21,
    "constraint": glll_to_int(GLLL.CHECK),  # 0x50
    "gauge": 0x23,
    # Rails/Gates aligned
    "GATE_B": glll_to_int(GLLL.GATE_B),  # 0x30
    "GATE_E": glll_to_int(GLLL.GATE_E),  # 0x31
    "CHECK": glll_to_int(GLLL.CHECK),     # 0x50
    "SEAL": glll_to_int(GLLL.SEAL),       # 0x3D
    "EMIT": glll_to_int(GLLL.EMIT),       # 0x38
    # Add any missing
}

class HadamardAssembler:
    def __init__(self):
        self.bytecode = bytearray()

    def add_instruction(self, opcode_str: str, arg1: int = 0, arg2: int = 0, meta: int = 0):
        if opcode_str not in HADAMARD_OPCODES:
            raise ValueError(f"Unknown opcode: {opcode_str}")
        opcode = HADAMARD_OPCODES[opcode_str]
        # Pack as 4 bytes: opcode, arg1, arg2, meta (all uint8)
        self.bytecode.extend(struct.pack('BBBB', opcode, arg1 & 0xFF, arg2 & 0xFF, meta & 0xFF))

    def get_bytecode(self) -> bytes:
        return bytes(self.bytecode)