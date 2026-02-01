"""
NSC-M3L Multi-Model Compilation to Bytecode

Compiles NSC-M3L programs to EXEC model bytecode for VM execution.
Implements compilation from AST through PIR to bytecode instructions.

Denotation: Program → Executable bytecode with execution semantics
"""

import hashlib
from typing import Dict, Set, List, Optional, Any, Tuple
from dataclasses import dataclass, field

# Import NSC components
from src.nsc.ast import (
    Program, Statement, Expr, Atom, BinaryOp, OpCall, Group,
    Equation, Functional, Constraint, Decl, Directive,
    Model, SemanticType
)
from src.nsc.exec_types import (
    OpCode, Instruction, BytecodeProgram, RuntimeValue, ValueType
)


# =============================================================================
# Operator to Opcode Mapping
# =============================================================================

# Mapping from operator names to bytecode opcodes
OPERATOR_OPCODES: Dict[str, OpCode] = {
    # Math operators
    "+": OpCode.ADD,
    "-": OpCode.SUB,
    "*": OpCode.MUL,
    "/": OpCode.DIV,
    "%": OpCode.MOD,
    "^": OpCode.POW,
    "neg": OpCode.NEG,
    "abs": OpCode.ABS,
    "sqrt": OpCode.SQRT,
    
    # NSC-M3L Physics operators
    "grad": OpCode.GRAD,
    "div": OpCode.DIV_OP,
    "curl": OpCode.CURL,
    "laplacian": OpCode.LAPLACIAN,
    "Δ": OpCode.LAPLACIAN,
    "∇": OpCode.GRAD,
    "d/dt": OpCode.DDT,
    "∂/∂t": OpCode.DDT,
    "partial": OpCode.PARTIAL,
    "∂": OpCode.PARTIAL,
    "cov": OpCode.COV_DERIV,
    "lie": OpCode.LIE_DERIV,
    
    # Geometry operators
    "hodge": OpCode.HODGE,
    "inner": OpCode.INNER,
    "wedge": OpCode.WEDGE,
    "contract": OpCode.CONTRACT,
    "sym": OpCode.SYM,
    "antisym": OpCode.ANTISYM,
    "trace": OpCode.TRACE,
    "lower": OpCode.LOWER,
    "raise": OpCode.RAISE,
    
    # Tensor construction
    "sym6": OpCode.MAKE_SYM6,
    "vec3": OpCode.MAKE_VEC3,
    
    # Comparison operators
    "==": OpCode.EQ,
    "!=": OpCode.NE,
    "<": OpCode.LT,
    "<=": OpCode.LE,
    ">": OpCode.GT,
    ">=": OpCode.GT,
    
    # Control flow
    "if": OpCode.BR_TRUE,
    "goto": OpCode.JMP,
    "call": OpCode.CALL,
    "return": OpCode.RET,
    "halt": OpCode.HALT,
    
    # GR operators
    "christoffel": OpCode.CHRISTOFFEL,
    "ricci": OpCode.RICCI,
    "ricci_scalar": OpCode.RICCI_SCALAR,
    "einstein": OpCode.EINSTEIN,
    "riemann": OpCode.RIEMANN,
    
    # BSSN operators
    "bssn_conform": OpCode.BSSN_CONFORM,
    "bssn_trace": OpCode.BSSN_TRACE,
    "bssn_gamma": OpCode.BSSN_GAMMA,
    "bssn_lambda": OpCode.BSSN_LAMBDA,
    
    # Ledger/EXEC operators
    "emit": OpCode.EMIT,
    "check_gate": OpCode.CHECK_GATE,
    "check_invariant": OpCode.CHECK_INV,
    "seal": OpCode.SEAL,
    
    # Constraint operators
    "enforce_H": OpCode.ENFORCE_H,
    "enforce_M": OpCode.ENFORCE_M,
    "project": OpCode.PROJECT,
    "phi": OpCode.PHI_FUNC,
}


# =============================================================================
# Compilation Context
# =============================================================================

@dataclass
class CompilationContext:
    """Context during compilation."""
    # Target models for compilation
    target_models: Set[Model] = field(default_factory=lambda: {Model.EXEC})
    
    # Symbol table
    variables: Dict[str, int] = field(default_factory=dict)  # name -> stack position
    constants: List[Any] = field(default_factory=list)
    fields: List[str] = field(default_factory=list)
    
    # Current compilation position
    instruction_index: int = 0
    source_map: Dict[int, str] = field(default_factory=dict)
    
    # Control flow
    label_positions: Dict[str, int] = field(default_factory=dict)
    jump_targets: List[Tuple[int, str]] = field(default_factory=list)  # (position, label)
    
    # Compilation options
    emit_source_map: bool = True
    optimize: bool = False
    
    def add_constant(self, value: Any) -> int:
        """Add constant and return its index."""
        if value in self.constants:
            return self.constants.index(value)
        self.constants.append(value)
        return len(self.constants) - 1
    
    def add_label(self, name: str) -> int:
        """Register label at current position."""
        self.label_positions[name] = self.instruction_index
        return self.instruction_index
    
    def add_jump_target(self, position: int, label: str):
        """Record jump target to resolve later."""
        self.jump_targets.append((position, label))
    
    def resolve_jumps(self, program: BytecodeProgram):
        """Resolve all jump targets."""
        for position, label in self.jump_targets:
            if label in self.label_positions:
                target = self.label_positions[label]
                if position < len(program.instructions):
                    program.instructions[position].immediate = target


# =============================================================================
# Exec Compiler
# =============================================================================

class ExecCompiler:
    """
    Compiler that transforms NSC-M3L programs to EXEC bytecode.
    
    Supports compilation from:
    - NSC AST (abstract syntax tree)
    - PIR (program intermediate representation)
    
    Produces:
    - BytecodeProgram with executable instructions
    - Source map for debugging
    """
    
    def __init__(self, target_models: Optional[Set[Model]] = None, 
                 emit_source_map: bool = True):
        """
        Initialize the compiler.
        
        Args:
            target_models: Models to target (default: {EXEC})
            emit_source_map: Whether to include source mapping
        """
        self.context = CompilationContext(
            target_models=target_models or {Model.EXEC},
            emit_source_map=emit_source_map
        )
    
    def compile(self, program: Program) -> BytecodeProgram:
        """
        Compile NSC-M3L program to bytecode.
        
        Args:
            program: NSC-M3L AST program
            
        Returns:
            BytecodeProgram ready for VM execution
        """
        bytecode = BytecodeProgram()
        
        # Build symbol table
        self._build_symbol_table(program)
        
        # Compile statements
        for stmt in program.statements:
            self._compile_statement(stmt, bytecode)
        
        # Resolve jumps
        self.context.resolve_jumps(bytecode)
        
        # Set metadata
        bytecode.constants = self.context.constants.copy()
        bytecode.source_map = self.context.source_map.copy()
        bytecode.entry_point = 0
        
        return bytecode
    
    def compile_ast(self, statements: List[Statement]) -> BytecodeProgram:
        """Compile from raw statements (no Program wrapper)."""
        program = Program(statements=statements)
        return self.compile(program)
    
    def _build_symbol_table(self, program: Program):
        """Build symbol table from program."""
        # Process declarations
        for stmt in program.statements:
            if isinstance(stmt, Decl):
                self.context.variables[stmt.ident] = len(self.context.variables)
            elif isinstance(stmt, Equation) and isinstance(stmt.lhs, Atom):
                self.context.variables[stmt.lhs.value] = len(self.context.variables)
    
    def _compile_statement(self, stmt: Statement, bytecode: BytecodeProgram):
        """Compile a single statement."""
        if isinstance(stmt, Equation):
            self._compile_equation(stmt, bytecode)
        elif isinstance(stmt, Functional):
            self._compile_functional(stmt, bytecode)
        elif isinstance(stmt, Constraint):
            self._compile_constraint(stmt, bytecode)
        elif isinstance(stmt, Decl):
            self._compile_declaration(stmt, bytecode)
        elif isinstance(stmt, Directive):
            self._compile_directive(stmt, bytecode)
    
    def _compile_equation(self, eq: Equation, bytecode: BytecodeProgram):
        """Compile equation statement (lhs = rhs)."""
        if eq.lhs is None or eq.rhs is None:
            return
        
        # Mark position for source map
        position = len(bytecode.instructions)
        if self.context.emit_source_map:
            self.context.source_map[position] = f"line_{eq.start}"
        
        # Compile RHS (value to compute)
        self._compile_expression(eq.rhs, bytecode)
        
        # Store result in LHS variable
        if isinstance(eq.lhs, Atom):
            var_name = eq.lhs.value
            if var_name in self.context.variables:
                var_idx = self.context.variables[var_name]
                bytecode.add_instruction(OpCode.STORE, operand1=var_idx)
        elif isinstance(eq.lhs, OpCall):
            # Handle operator assignment (e.g., "grad(f) = ...")
            self._compile_expression(eq.lhs, bytecode)
    
    def _compile_functional(self, func: Functional, bytecode: BytecodeProgram):
        """Compile functional definition."""
        # For now, just emit PUSH with metadata
        bytecode.add_instruction(OpCode.NOP)
    
    def _compile_constraint(self, cons: Constraint, bytecode: BytecodeProgram):
        """Compile constraint definition."""
        if cons.predicate is not None:
            self._compile_expression(cons.predicate, bytecode)
        
        # Emit invariant check if name is provided
        if cons.ident:
            bytecode.add_instruction(OpCode.CHECK_INV, immediate=len(cons.ident))
    
    def _compile_declaration(self, decl: Decl, bytecode: BytecodeProgram):
        """Compile variable declaration."""
        # For stack-based VM, declarations don't emit code
        # The variable is added to symbol table during preprocessing
        pass
    
    def _compile_directive(self, directive: Directive, bytecode: BytecodeProgram):
        """Compile directive statement."""
        # Directives are metadata, may emit instructions for target selection
        if directive.directive_type.value == "compile":
            # Handle compilation target directive
            bytecode.add_instruction(OpCode.NOP)
    
    def _compile_expression(self, expr: Expr, bytecode: BytecodeProgram) -> int:
        """
        Compile expression to bytecode instructions.
        
        Returns:
            Number of values left on stack
        """
        if isinstance(expr, Atom):
            return self._compile_atom(expr, bytecode)
        elif isinstance(expr, BinaryOp):
            return self._compile_binary_op(expr, bytecode)
        elif isinstance(expr, OpCall):
            return self._compile_opcall(expr, bytecode)
        elif isinstance(expr, Group):
            return self._compile_expression(expr.inner, bytecode) if expr.inner else 0
        else:
            return 0
    
    def _compile_atom(self, atom: Atom, bytecode: BytecodeProgram) -> int:
        """Compile atomic expression (number, variable, etc.)."""
        value = atom.value
        
        # Check if it's a number
        try:
            num_value = float(value)
            const_idx = self.context.add_constant(num_value)
            bytecode.add_instruction(OpCode.LOAD_CONST, immediate=const_idx)
            return 1
        except ValueError:
            pass
        
        # Check if it's a variable
        if value in self.context.variables:
            var_idx = self.context.variables[value]
            bytecode.add_instruction(OpCode.LOAD, operand1=var_idx)
            return 1
        
        # Unknown identifier - push zero
        bytecode.add_instruction(OpCode.PUSH, immediate=0)
        return 1
    
    def _compile_binary_op(self, op: BinaryOp, bytecode: BytecodeProgram) -> int:
        """Compile binary operation."""
        # Compile operands first (post-order traversal)
        left_count = self._compile_expression(op.left, bytecode)
        right_count = self._compile_expression(op.right, bytecode)
        
        # Get opcode for operator
        opcode = OPERATOR_OPCODES.get(op.op)
        if opcode is None:
            # Try to find by symbol
            for name, op_code in OPERATOR_OPCODES.items():
                if op_code.name.lower() == op.op.lower():
                    opcode = op_code
                    break
        
        if opcode is not None:
            bytecode.add_instruction(opcode)
        
        # Return result count (1 value on stack)
        return max(left_count, right_count) - 1
    
    def _compile_opcall(self, opcall: OpCall, bytecode: BytecodeProgram) -> int:
        """Compile operator call (e.g., grad(f), div(v))."""
        # Compile arguments first
        arg_count = 0
        if opcall.arg is not None:
            arg_count = self._compile_expression(opcall.arg, bytecode)
        
        # Get opcode for operator
        op_name = opcall.op
        
        # Try various name forms
        opcode = OPERATOR_OPCODES.get(op_name)
        if opcode is None:
            # Try with common prefixes/suffixes
            alt_names = [
                op_name.replace("grad", "∇"),
                op_name.replace("∇", "grad"),
                op_name.replace("div", "∇·"),
                op_name.replace("laplacian", "Δ"),
                op_name.replace("Δ", "laplacian"),
            ]
            for alt in alt_names:
                if alt in OPERATOR_OPCODES:
                    opcode = OPERATOR_OPCODES[alt]
                    break
        
        if opcode is not None:
            bytecode.add_instruction(opcode)
        
        # Result depends on operator
        return self._get_operator_result_count(op_name, arg_count)
    
    def _get_operator_result_count(self, op_name: str, arg_count: int) -> int:
        """Get how many values an operator leaves on stack."""
        scalar_ops = {"grad", "div", "curl", "laplacian", "∂/∂t", "d/dt", "partial", 
                      "trace", "ricci_scalar", "enforce_H", "enforce_M"}
        tensor_ops = {"∇", "Δ", "cov", "lie", "hodge", "inner", "wedge", "contract",
                      "christoffel", "ricci", "einstein", "riemann"}
        
        op_lower = op_name.lower()
        for scalar_op in scalar_ops:
            if scalar_op.lower() in op_lower:
                return 1
        for tensor_op in tensor_ops:
            if tensor_op.lower() in op_lower:
                return 1
        
        # Default: binary ops leave 1 result, unary ops leave 1
        return 1


# =============================================================================
# PIR to Bytecode Compilation
# =============================================================================

@dataclass
class PIRNode:
    """PIR node for intermediate representation."""
    op: str
    args: List['PIRNode'] = field(default_factory=list)
    result_idx: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class PIRToBytecodeCompiler:
    """
    Compiler from PIR (Program Intermediate Representation) to bytecode.
    
    PIR is a simpler linear IR that's easier to generate from parsers.
    """
    
    def __init__(self):
        self.context = CompilationContext()
    
    def compile_pir(self, pir_nodes: List[PIRNode]) -> BytecodeProgram:
        """Compile PIR nodes to bytecode."""
        bytecode = BytecodeProgram()
        
        for node in pir_nodes:
            self._compile_pir_node(node, bytecode)
        
        return bytecode
    
    def _compile_pir_node(self, node: PIRNode, bytecode: BytecodeProgram):
        """Compile a single PIR node."""
        # First compile arguments
        for arg in node.args:
            self._compile_pir_node(arg, bytecode)
        
        # Then emit operator
        opcode = OPERATOR_OPCODES.get(node.op)
        if opcode is not None:
            bytecode.add_instruction(opcode)


# =============================================================================
# Convenience Functions
# =============================================================================

def compile_nsc_to_bytecode(source: str) -> BytecodeProgram:
    """
    Convenience function to compile NSC source to bytecode.
    
    Args:
        source: NSC-M3L source code
        
    Returns:
        BytecodeProgram
    """
    # Import here to avoid circular imports
    from src.nsc.parse import parse_program
    
    tokens = []  # Would need lexer here
    program = parse_program(tokens)  # Simplified
    return ExecCompiler().compile(program)


def compile_program_to_exec(program: Program) -> BytecodeProgram:
    """
    Compile Program AST to EXEC bytecode.
    
    Args:
        program: NSC-M3L Program AST
        
    Returns:
        BytecodeProgram ready for VM execution
    """
    return ExecCompiler().compile(program)


def compile_ast_to_exec(statements: List[Statement]) -> BytecodeProgram:
    """
    Compile statements to EXEC bytecode.
    
    Args:
        statements: List of statement AST nodes
        
    Returns:
        BytecodeProgram
    """
    return ExecCompiler().compile_ast(statements)


# =============================================================================
# Demo/Test
# =============================================================================

if __name__ == '__main__':
    # Create simple program for testing
    from src.nsc.ast import Program, Equation, Atom
    
    # Create test program
    stmts = [
        Equation(lhs=Atom(value="x", start=0, end=1), 
                 rhs=Atom(value="2", start=4, end=5),
                 start=0, end=6),
        Equation(lhs=Atom(value="y", start=8, end=9),
                 rhs=BinaryOp(op="+", left=Atom(value="x", start=12, end=13),
                              right=Atom(value="3", start=16, end=17),
                              start=12, end=17),
                 start=8, end=18),
    ]
    
    program = Program(statements=stmts)
    
    # Compile
    compiler = ExecCompiler()
    bytecode = compiler.compile(program)
    
    print(f"Compiled {len(stmts)} statements to {len(bytecode.instructions)} instructions")
    print(f"Constants: {bytecode.constants}")
    print(f"Instructions:")
    for i, inst in enumerate(bytecode.instructions):
        print(f"  {i}: {inst}")
