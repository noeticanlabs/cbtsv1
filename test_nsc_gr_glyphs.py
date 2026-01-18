#!/usr/bin/env python3
"""Test NSC_GR glyph extensions."""

from noetica_nsc_phase1.nsc import tokenize, nsc_to_pde, GLYPH_TO_OPCODE

def test_gr_glyphs():
    # Test the new GR glyphs
    test_source = "ℋ 𝓜 𝔊 𝔇 𝔅 𝔄 𝔯 𝕋"
    tokens = tokenize(test_source)
    print("Tokens:", tokens)

    prog, flattened, bytecode, pde = nsc_to_pde(test_source)
    print("Flattened:", flattened)
    print("Bytecode opcodes:", bytecode.opcodes)
    print("Expected opcodes:", [GLYPH_TO_OPCODE[g] for g in ['ℋ', '𝓜', '𝔊', '𝔇', '𝔅', '𝔄', '𝔯', '𝕋']])

    assert bytecode.opcodes == [0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27, 0x28], f"Opcode mismatch: {bytecode.opcodes}"
    print("NSC_GR glyphs test passed!")

if __name__ == "__main__":
    test_gr_glyphs()