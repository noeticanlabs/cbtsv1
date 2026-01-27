"""
NSC to Hadamard Pipeline Integration

This module bridges the NSC compiler's PIR (Program Intermediate Representation)
to Hadamard bytecode, enabling end-to-end compilation from .nsc files to
executable bytecode.

Pipeline:
    .nsc file → NSC Parser → Flatten → Assemble PDE → PIR
                                      ↓
                        NSCToHadamardCompiler
                                      ↓
                        Hadamard Bytecode (4-byte format)
"""

import struct
from typing import Dict, List, Optional, Tuple, Any

import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.nsc.parse import parse_program
from src.nsc.lower_pir import lower_to_pir
from src.solver.pir import (
    PIRProgram, Operator, Field, Boundary, Integrator, StepLoop,
    EffectSignature, GR_OP_SIGNATURES
)
from src.hadamard.assembler import HadamardAssembler, glll_to_int
from src.triaxis.lexicon import GLLL


# =============================================================================
# PIR → Hadamard Opcode Mapping
# =============================================================================

# Mapping of PIR operator types to Hadamard opcode strings
# The assembler will resolve these to actual bytecodes
PIR_TO_HADAMARD_OPCODES: Dict[str, str] = {
    # Core GR operators - use glyphs that assembler knows about
    'diffusion': '∆',            # Diffusion operator (damping glyph)
    'source': '⊕',               # Source term (add glyph)
    'sink': '⊖',                 # Sink term (subtract glyph)
    'curvature_coupling': '↻',   # Christoffel symbol coupling
    'damping': '∆',              # Damping operator
    'gauge_enforcement': '⊕',    # Gauge enforcement (reuse add)
    'dissipation': '∆',          # Dissipation via diffusion
    
    # Control flow
    'evolve': '⇒',               # Step emission (arrow glyph)
    'solve_constraints': '⊕',    # Constraint verification
    'update_geometry': '⊕',      # Geometry update
    
    # Boundary handling
    'dirichlet': '□',            # Dirichlet boundary marker
    'neumann': '𝔅',              # Neumann boundary marker
}

# GLLL opcode constants for reference
HADAMARD_OPCODE_MAP = {
    'NOP': 0x00,
    'HALT': 0x01,
    'ADD': glll_to_int(GLLL.ADD),      # 0x32
    'SUB': glll_to_int(GLLL.SUB),      # 0x33
    'MUL': glll_to_int(GLLL.MUL),      # 0x34
    'DIV': glll_to_int(GLLL.DIV),      # 0x35
    'MOV': glll_to_int(GLLL.MOV),      # 0x18
    'STORE': glll_to_int(GLLL.STORE),  # 0x17
    'LOAD': glll_to_int(GLLL.LOAD),    # 0x16
    'PUSH': glll_to_int(GLLL.PUSH),    # 0x22
    'POP': glll_to_int(GLLL.POP),      # 0x23
    'GATE_B': glll_to_int(GLLL.GATE_B),  # 0x30
    'GATE_E': glll_to_int(GLLL.GATE_E),  # 0x31
    'CHECK': glll_to_int(GLLL.CHECK),    # 0x50
    'EMIT': glll_to_int(GLLL.EMIT),      # 0x38
    'DIFFUSE': glll_to_int(GLLL.DIFFUSE),  # H128:r69
    'LAPLACE': glll_to_int(GLLL.LAPLACE),  # H128:r72
    'CONNECTION_COEFF': glll_to_int(GLLL.CONNECTION_COEFF),  # H128:r73
}


# =============================================================================
# Field Mapping
# =============================================================================

# NSC field names to Hadamard field indices
# Ordered for deterministic execution
DEFAULT_FIELD_ORDER = ['theta', 'phi', 'gamma', 'alpha', 'beta', 'K']

# Mapping from NSC field names to canonical field indices
FIELD_NAME_TO_INDEX: Dict[str, int] = {
    'theta': 0,   # Conformal factor
    'phi': 1,     # Gravitational potential
    'gamma': 2,   # Spatial metric
    'alpha': 3,   # Lapse function
    'beta': 4,    # Shift vector
    'K': 5,       # Extrinsic curvature
}


def get_field_index(field_name: str, custom_order: Optional[List[str]] = None) -> int:
    """
    Get the Hadamard field index for a given NSC field name.
    
    Args:
        field_name: Name of the field (e.g., 'theta', 'phi')
        custom_order: Optional custom field order for deterministic mapping
    
    Returns:
        Field index (0-based) for Hadamard bytecode
    """
    if custom_order is not None:
        try:
            return custom_order.index(field_name)
        except ValueError:
            # Field not in custom order, append at end
            return len(custom_order)
    
    return FIELD_NAME_TO_INDEX.get(field_name, 0)


def build_field_mapping(fields: List[Field], custom_order: Optional[List[str]] = None) -> Dict[str, int]:
    """
    Build a complete field mapping from NSC fields.
    
    Args:
        fields: List of Field objects from PIR
        custom_order: Optional custom field order
    
    Returns:
        Dictionary mapping field names to Hadamard indices
    """
    mapping: Dict[str, int] = {}
    
    if custom_order is not None:
        # Use custom order
        for idx, field_name in enumerate(custom_order):
            if any(f.name == field_name for f in fields):
                mapping[field_name] = idx
    else:
        # Use default mapping
        for field in fields:
            mapping[field.name] = get_field_index(field.name)
    
    return mapping


# =============================================================================
# NSCToHadamardCompiler
# =============================================================================

class NSCToHadamardCompiler:
    """
    Compiler that transforms NSC programs (via PIR) into Hadamard bytecode.
    
    This is the bridge between the NSC compiler's abstract representation
    and the Hadamard VM's executable bytecode format.
    
    Attributes:
        field_order: Custom field ordering for deterministic execution
        trace_enabled: Whether to include trace metadata in bytecode
    """
    
    def __init__(self, field_order: Optional[List[str]] = None, trace_enabled: bool = False):
        """
        Initialize the compiler.
        
        Args:
            field_order: Optional custom field order for deterministic mapping
            trace_enabled: Whether to emit trace information in bytecode
        """
        self.field_order = field_order or DEFAULT_FIELD_ORDER.copy()
        self.trace_enabled = trace_enabled
    
    def compile_pir(self, pir_program: PIRProgram) -> bytes:
        """
        Compile a PIR program to Hadamard bytecode.
        
        Args:
            pir_program: The PIR representation of the NSC program
        
        Returns:
            Hadamard bytecode (4-byte instruction format)
        """
        assembler = HadamardAssembler()
        
        # Build field mapping for this program
        field_mapping = build_field_mapping(pir_program.fields, self.field_order)
        
        # Emit header with program metadata
        self._emit_header(assembler, pir_program, field_mapping)
        
        # Emit field declarations
        self._emit_field_declarations(assembler, pir_program.fields, field_mapping)
        
        # Emit operators in order
        for operator in pir_program.operators:
            self._emit_operator(assembler, operator, field_mapping)
        
        # Emit boundary specification
        self._emit_boundary(assembler, pir_program.boundary)
        
        # Emit integrator setup
        self._emit_integrator(assembler, pir_program.integrator)
        
        # Emit step loop
        self._emit_step_loop(assembler, pir_program.step_loop)
        
        return assembler.get_bytecode()
    
    def compile_nsc_file(self, path: str) -> bytes:
        """
        Compile an .nsc source file directly to Hadamard bytecode.
        
        Args:
            path: Path to the .nsc source file
        
        Returns:
            Hadamard bytecode
        """
        # Read and parse the source file
        with open(path, 'r', encoding='utf-8') as f:
            source = f.read()
        
        return self.compile_nsc_source(source)
    
    def compile_nsc_source(self, source: str) -> bytes:
        """
        Compile NSC source code to Hadamard bytecode.
        
        Args:
            source: NSC source code as string
        
        Returns:
            Hadamard bytecode
        """
        # Lex the source
        from src.nsc.lex import tokenize
        tokens = tokenize(source)
        
        # Parse to AST
        program = parse_program(tokens)
        
        # Lower to PIR
        pir_program = lower_to_pir(program)
        
        # Compile PIR to bytecode
        return self.compile_pir(pir_program)
    
    def _emit_header(self, assembler: HadamardAssembler, 
                     pir_program: PIRProgram, field_mapping: Dict[str, int]):
        """Emit program header with metadata."""
        # Magic number for Hadamard bytecode (4 bytes)
        assembler.add_instruction('*', 0, 0, 0)  # Placeholder using MUL as NOP
        
        # Field count in metadata
        field_count = len(pir_program.fields)
        assembler.add_instruction('+', field_count, 0, 0)
        
        # Operator count
        op_count = len(pir_program.operators)
        assembler.add_instruction('+', op_count, 0, 0)
    
    def _emit_field_declarations(self, assembler: HadamardAssembler,
                                  fields: List[Field], field_mapping: Dict[str, int]):
        """Emit field declarations for each field in the program."""
        for field in fields:
            field_idx = field_mapping.get(field.name, 0)
            field_type = 0 if field.type == 'scalar' else 1  # 0=scalar, 1=array
            
            # Use ADD to emit field declaration
            assembler.add_instruction('+', field_idx, field_type, 0)
    
    def _emit_operator(self, assembler: HadamardAssembler,
                       operator: Operator, field_mapping: Dict[str, int]):
        """Emit bytecode for a single PIR operator."""
        op_type = operator.type
        
        # Get effect signature for metadata
        effect_sig = operator.effect_signature
        
        if op_type in PIR_TO_HADAMARD_OPCODES:
            opcode_str = PIR_TO_HADAMARD_OPCODES[op_type]
        else:
            # Unknown operator, use NOP with error marker
            opcode_str = 'NOP'
        
        # Encode effect signature in metadata byte
        meta = self._encode_effect_signature(effect_sig)
        
        # Emit the instruction
        if opcode_str in HADAMARD_OPCODE_MAP:
            # Use direct bytecode for known GLLL ops
            assembler.add_instruction(opcode_str, 0, 0, meta)
        else:
            # Use assembler's generic add_instruction
            assembler.add_instruction(opcode_str, 0, 0, meta)
        
        # Emit trace information if enabled
        if self.trace_enabled and operator.trace:
            for trace_entry in operator.trace:
                self._emit_trace_entry(assembler, trace_entry)
    
    def _emit_boundary(self, assembler: HadamardAssembler, boundary: Boundary):
        """Emit boundary specification."""
        if boundary.type == 'dirichlet':
            assembler.add_instruction('□', 0, 0, 0)  # Dirichlet marker
        elif boundary.type == 'neumann':
            assembler.add_instruction('𝔅', 0, 0, 0)  # Neumann marker
        # 'none' emits nothing
    
    def _emit_integrator(self, assembler: HadamardAssembler, integrator: Integrator):
        """Emit integrator configuration."""
        if integrator.type == 'RK2':
            assembler.add_instruction('⇒', 2, 0, 0)  # RK2 marker
        else:
            # Default to Euler (emit nothing special, use default)
            assembler.add_instruction('*', 0, 0, 0)
    
    def _emit_step_loop(self, assembler: HadamardAssembler, step_loop: StepLoop):
        """Emit step loop configuration."""
        # Emit N (step count) and dt (timestep)
        N_low = step_loop.N & 0xFF
        N_high = (step_loop.N >> 8) & 0xFF
        
        # Convert dt to fixed-point for bytecode
        dt_fixed = int(step_loop.dt * 1000) & 0xFF  # 3 decimal places
        
        assembler.add_instruction('+', N_low, N_high, dt_fixed)
        assembler.add_instruction('*', 0, 0, 0)
    
    def _encode_effect_signature(self, effect_sig: EffectSignature) -> int:
        """
        Encode effect signature into a single metadata byte.
        
        Byte format:
            bits 0-1: determinism (00=pure, 01=reads_only, 10=writes)
            bit 2: requires_audit_before_commit
            bits 3-7: reserved
        """
        determinism_map = {'pure': 0, 'reads_only': 1, 'writes': 2}
        det_val = determinism_map.get(effect_sig.determinism, 0)
        
        audit_bit = 1 if effect_sig.requires_audit_before_commit else 0
        
        return (det_val & 0x03) | ((audit_bit & 0x01) << 2)
    
    def _emit_trace_entry(self, assembler: HadamardAssembler, trace_entry):
        """Emit trace entry for debugging."""
        # Encode glyph as opcode, origin info in args
        glyph_ord = ord(trace_entry.glyph) if len(trace_entry.glyph) == 1 else 0
        assembler.add_instruction('*', glyph_ord, 0, 0)


# =============================================================================
# Convenience Functions
# =============================================================================

def compile_nsc_to_hadamard(source: str, field_order: Optional[List[str]] = None) -> bytes:
    """
    Convenience function to compile NSC source to Hadamard bytecode.
    
    Args:
        source: NSC source code
        field_order: Optional custom field order
    
    Returns:
        Hadamard bytecode
    """
    compiler = NSCToHadamardCompiler(field_order=field_order)
    return compiler.compile_nsc_source(source)


def compile_nsc_file_to_hadamard(path: str, field_order: Optional[List[str]] = None) -> bytes:
    """
    Convenience function to compile an .nsc file to Hadamard bytecode.
    
    Args:
        path: Path to .nsc file
        field_order: Optional custom field order
    
    Returns:
        Hadamard bytecode
    """
    compiler = NSCToHadamardCompiler(field_order=field_order)
    return compiler.compile_nsc_file(path)


# =============================================================================
# Demo / Test
# =============================================================================

if __name__ == '__main__':
    # Example NSC source
    example_nsc = """
    ◯ ⊕ (↻)
    ⇒
    """
    
    # Compile to bytecode
    compiler = NSCToHadamardCompiler()
    bytecode = compiler.compile_nsc_source(example_nsc)
    
    print(f"Compiled {len(example_nsc)} chars to {len(bytecode)} bytes of bytecode")
    print(f"Bytecode (hex): {bytecode.hex()}")
    print(f"Bytecode (raw): {list(bytecode)}")
