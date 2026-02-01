"""
Virtual Machine Core for NSC-M3L Bytecode Execution

Implements stack-based execution with physics/geometry operations.
"""

import time
import hashlib
import numpy as np
from typing import List, Any, Dict, Optional, Tuple, Callable
from dataclasses import dataclass, field

# Import NSC types
from src.nsc.exec_types import (
    OpCode, Instruction, BytecodeProgram, RuntimeValue, ValueType,
    ExecutionState, Receipt
)

# Import submodules
from src.nsc.exec_vm.errors import (
    VMError, StackUnderflowError, StackOverflowError,
    DivisionByZeroError, GateViolationError, InvariantViolationError
)
from src.nsc.exec_vm.physics_kernels import (
    _compute_gradient_jit, _compute_divergence_jit,
    _compute_laplacian_jit, _compute_curl_jit
)


@dataclass
class VirtualMachine:
    """
    NSC-M3L Virtual Machine for bytecode execution.
    
    Features:
        - Stack-based instruction execution
        - Register file for fast access
        - Memory management for fields
        - Deterministic scheduling
        - Stage-time protocol support
        - Ledger integration for receipts
        - JIT compilation for hot loops
    """
    
    program: BytecodeProgram
    stack_size: int = 1024
    num_registers: int = 64
    
    # Runtime state (not in constructor to allow reset)
    _ip: int = 0
    _sp: int = 0
    _fp: int = 0  # Frame pointer
    
    # Execution resources
    stack: List[RuntimeValue] = field(default_factory=list)
    memory: Dict[int, RuntimeValue] = field(default_factory=dict)
    registers: List[RuntimeValue] = field(default_factory=list)
    
    # Field storage
    fields: Dict[str, np.ndarray] = field(default_factory=dict)
    
    # Execution control
    state: str = "idle"  # idle, running, halted, error, yielding
    step_number: int = 0
    max_steps: int = 100000
    
    # Timing
    clocks: Dict[str, float] = field(default_factory=dict)
    stage_times: Dict[str, float] = field(default_factory=dict)
    
    # Ledger integration
    receipts: List[Receipt] = field(default_factory=list)
    receipt_counter: int = 0
    
    # Determinism
    operation_log: List[Dict] = field(default_factory=list)
    
    # Gate/invariant checking
    gates: Dict[int, float] = field(default_factory=dict)
    invariants: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    
    # Call stack for functions
    call_stack: List[int] = field(default_factory=list)
    frame_stack: List[int] = field(default_factory=list)
    
    # JIT cache
    jit_cache: Dict[int, Callable] = field(default_factory=dict)
    jit_tier: int = 0
    
    def __post_init__(self):
        """Initialize VM state."""
        self.registers = [RuntimeValue(0, ValueType.INT) for _ in range(self.num_registers)]
        self._reset()
    
    def _reset(self):
        """Reset VM to initial state."""
        self._ip = self.program.entry_point
        self._sp = -1  # Stack grows up, -1 = empty
        self._fp = -1
        self.stack = []
        self.memory = {}
        self.registers = [RuntimeValue(0, ValueType.INT) for _ in range(self.num_registers)]
        self.state = "idle"
        self.step_number = 0
        self.receipts = []
        self.receipt_counter = 0
        self.operation_log = []
        self.call_stack = []
        self.frame_stack = []
    
    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------
    
    def run(self, max_steps: Optional[int] = None) -> Tuple[str, Any]:
        """
        Execute the program.
        
        Args:
            max_steps: Maximum number of instructions to execute
            
        Returns:
            Tuple of (final_state, result_value)
        """
        if max_steps is not None:
            self.max_steps = max_steps
        
        self._reset()
        self.state = "running"
        
        start_time = time.time()
        self.clocks['execution_start'] = start_time
        
        try:
            for step in range(self.max_steps):
                if self.state != "running":
                    break
                
                # Fetch and execute
                inst = self._fetch()
                if inst is None:
                    break
                self._execute(inst)
                self.step_number += 1
                
                # Check for JIT compilation
                if self._ip in self.jit_cache:
                    self._execute_jit(self._ip)
            
            self.state = "halted" if self.state == "running" else self.state
            
        except VMError as e:
            self.state = "error"
            raise e
        
        self.clocks['execution_end'] = time.time()
        self.clocks['execution_time'] = self.clocks['execution_end'] - self.clocks['execution_start']
        
        return self.state, self._get_result()
    
    def step(self) -> Tuple[str, Any]:
        """
        Execute a single step.
        
        Returns:
            Tuple of (new_state, result_value)
        """
        if self.state not in ("idle", "running"):
            return self.state, None
        
        self.state = "running"
        inst = self._fetch()
        
        if inst is None:
            self.state = "halted"
            return self.state, None
        
        self._execute(inst)
        self.step_number += 1
        
        return self.state, self._get_result()
    
    def set_field(self, name: str, data: np.ndarray):
        """Set a named field for computation."""
        self.fields[name] = data
    
    def get_field(self, name: str) -> Optional[np.ndarray]:
        """Get a named field."""
        return self.fields.get(name)
    
    def set_gate_threshold(self, gate_id: int, threshold: float):
        """Set gate threshold for checking."""
        self.gates[gate_id] = threshold
    
    def set_invariant(self, inv_id: str, expected: float, tolerance: float = 1e-10):
        """Set invariant for checking."""
        self.invariants[inv_id] = (expected, tolerance)
    
    # -------------------------------------------------------------------------
    # Instruction Fetch/Execute
    # -------------------------------------------------------------------------
    
    def _fetch(self) -> Optional[Instruction]:
        """Fetch next instruction."""
        if self._ip >= len(self.program.instructions):
            self.state = "halted"
            return None
        
        inst = self.program.instructions[self._ip]
        self._ip += 1
        return inst
    
    def _execute(self, inst: Instruction):
        """Execute a single instruction."""
        # Log operation for determinism verification
        self.operation_log.append({
            'step': self.step_number,
            'ip': self._ip - 1,
            'opcode': inst.opcode.name,
            'timestamp': time.time()
        })
        
        # Dispatch to handler
        handler = self._get_handler(inst.opcode)
        try:
            handler(inst)
        except ZeroDivisionError:
            raise DivisionByZeroError(self._ip - 1)
        except IndexError:
            raise StackUnderflowError(self._ip - 1)
    
    def _get_handler(self, opcode: OpCode) -> Callable[[Instruction], None]:
        """Get handler function for opcode."""
        handler_name = f'_handle_{opcode.name.lower()}'
        handler = getattr(self, handler_name, self._handle_unknown)
        return handler
    
    def _handle_unknown(self, inst: Instruction):
        """Handle unknown opcode."""
        raise VMError(f"Unknown opcode: {inst.opcode}", self._ip - 1, inst)
    
    # -------------------------------------------------------------------------
    # Stack Operations
    # -------------------------------------------------------------------------
    
    def _push(self, value: Any, value_type: ValueType = ValueType.FLOAT):
        """Push value onto stack."""
        if len(self.stack) >= self.stack_size:
            raise StackOverflowError(self._ip)
        
        rv = RuntimeValue(value, value_type)
        self.stack.append(rv)
        self._sp = len(self.stack) - 1
    
    def _pop(self) -> RuntimeValue:
        """Pop value from stack."""
        if not self.stack:
            raise StackUnderflowError(self._ip)
        
        rv = self.stack.pop()
        self._sp = len(self.stack) - 1
        return rv
    
    def _peek(self, offset: int = 0) -> RuntimeValue:
        """Peek at stack value."""
        idx = len(self.stack) - 1 - offset
        if idx < 0:
            raise StackUnderflowError(self._ip)
        return self.stack[idx]
    
    def _pop_number(self) -> float:
        """Pop numeric value from stack."""
        rv = self._pop()
        return float(rv.value)
    
    def _pop_int(self) -> int:
        """Pop integer value from stack."""
        rv = self._pop()
        return int(rv.value)
    
    # -------------------------------------------------------------------------
    # Register Operations
    # -------------------------------------------------------------------------
    
    def _load_reg(self, idx: int) -> RuntimeValue:
        """Load from register."""
        if 0 <= idx < len(self.registers):
            return self.registers[idx]
        return RuntimeValue(0, ValueType.INT)
    
    def _store_reg(self, idx: int, value: RuntimeValue):
        """Store to register."""
        if 0 <= idx < len(self.registers):
            self.registers[idx] = value
    
    # -------------------------------------------------------------------------
    # Opcode Handlers - Control Flow
    # -------------------------------------------------------------------------
    
    def _handle_nop(self, inst: Instruction):
        """No operation."""
        pass
    
    def _handle_halt(self, inst: Instruction):
        """Halt execution."""
        self.state = "halted"
    
    def _handle_jmp(self, inst: Instruction):
        """Unconditional jump."""
        target = inst.immediate
        if 0 <= target < len(self.program.instructions):
            self._ip = target
    
    def _handle_br(self, inst: Instruction):
        """Conditional branch (pop condition, jump if true)."""
        cond = self._pop_int()
        target = inst.immediate
        if cond != 0:
            self._ip = target
    
    def _handle_br_true(self, inst: Instruction):
        """Branch if true (non-zero)."""
        cond = self._pop_int()
        target = inst.immediate
        if cond:
            self._ip = target
    
    def _handle_br_false(self, inst: Instruction):
        """Branch if false (zero)."""
        cond = self._pop_int()
        target = inst.immediate
        if not cond:
            self._ip = target
    
    def _handle_call(self, inst: Instruction):
        """Function call."""
        target = inst.immediate
        self.call_stack.append(self._ip)
        self._ip = target
    
    def _handle_ret(self, inst: Instruction):
        """Return from function call."""
        if self.call_stack:
            self._ip = self.call_stack.pop()
        else:
            self.state = "halted"
    
    def _handle_yield(self, inst: Instruction):
        """Yield execution (for stage transitions)."""
        self.state = "yielding"
        self.stage_times[f'step_{self.step_number}'] = time.time()
    
    def _handle_await(self, inst: Instruction):
        """Await stage completion (resume from yield)."""
        if self.state == "yielding":
            self.state = "running"
    
    # -------------------------------------------------------------------------
    # Opcode Handlers - Stack Operations
    # -------------------------------------------------------------------------
    
    def _handle_push(self, inst: Instruction):
        """Push immediate value onto stack."""
        self._push(inst.immediate, ValueType.FLOAT)
    
    def _handle_pop(self, inst: Instruction):
        """Pop from stack."""
        self._pop()
    
    def _handle_dup(self, inst: Instruction):
        """Duplicate top of stack."""
        rv = self._peek()
        self.stack.append(rv)
    
    def _handle_swap(self, inst: Instruction):
        """Swap top two stack values."""
        if len(self.stack) >= 2:
            self.stack[-1], self.stack[-2] = self.stack[-2], self.stack[-1]
    
    def _handle_rot(self, inst: Instruction):
        """Rotate top three stack values."""
        if len(self.stack) >= 3:
            a, b, c = self.stack[-3], self.stack[-2], self.stack[-1]
            self.stack[-3], self.stack[-2], self.stack[-1] = b, c, a
    
    def _handle_over(self, inst: Instruction):
        """Push copy of second-to-top."""
        if len(self.stack) >= 2:
            self.stack.append(self.stack[-2])
    
    def _handle_pick(self, inst: Instruction):
        """Pick nth from top."""
        n = inst.operand1
        if len(self.stack) > n:
            self.stack.append(self.stack[-1 - n])
    
    # -------------------------------------------------------------------------
    # Opcode Handlers - Memory Operations
    # -------------------------------------------------------------------------
    
    def _handle_load(self, inst: Instruction):
        """Load from register."""
        idx = inst.operand1
        self._push(self._load_reg(idx).value)
    
    def _handle_store(self, inst: Instruction):
        """Store to register."""
        idx = inst.operand1
        rv = self._pop()
        self._store_reg(idx, rv)
    
    def _handle_alloc(self, inst: Instruction):
        """Allocate memory."""
        size = inst.immediate
        addr = hash(f"alloc_{self.step_number}_{size}") % (1 << 20)
        self._push(addr, ValueType.INT)
    
    def _handle_free(self, inst: Instruction):
        """Free memory."""
        addr = self._pop_int()
    
    def _handle_load_const(self, inst: Instruction):
        """Load constant from constant pool."""
        const_idx = inst.immediate
        if const_idx < len(self.program.constants):
            const = self.program.constants[const_idx]
            const_type = ValueType.FLOAT if isinstance(const, (int, float)) else ValueType.INT
            self._push(const, const_type)
    
    def _handle_load_field(self, inst: Instruction):
        """Load field by name."""
        field_idx = inst.operand1
        field_names = list(self.fields.keys())
        if 0 <= field_idx < len(field_names):
            name = field_names[field_idx]
            if name in self.fields:
                self._push(name, ValueType.FIELD)
    
    def _handle_store_field(self, inst: Instruction):
        """Store to named field."""
        name = self._pop().value
        value = self._pop()
    
    # -------------------------------------------------------------------------
    # Opcode Handlers - Math Operations
    # -------------------------------------------------------------------------
    
    def _handle_add(self, inst: Instruction):
        """Add: b + a (stack: ..., a, b → ..., a+b)"""
        b = self._pop_number()
        a = self._pop_number()
        self._push(a + b)
    
    def _handle_sub(self, inst: Instruction):
        """Subtract: b - a"""
        b = self._pop_number()
        a = self._pop_number()
        self._push(a - b)
    
    def _handle_mul(self, inst: Instruction):
        """Multiply."""
        b = self._pop_number()
        a = self._pop_number()
        self._push(a * b)
    
    def _handle_div(self, inst: Instruction):
        """Divide."""
        b = self._pop_number()
        a = self._pop_number()
        if b == 0:
            raise DivisionByZeroError(self._ip)
        self._push(a / b)
    
    def _handle_mod(self, inst: Instruction):
        """Modulo."""
        b = self._pop_number()
        a = self._pop_number()
        self._push(a % b)
    
    def _handle_neg(self, inst: Instruction):
        """Negate."""
        a = self._pop_number()
        self._push(-a)
    
    def _handle_abs(self, inst: Instruction):
        """Absolute value."""
        a = self._pop_number()
        self._push(abs(a))
    
    def _handle_sqrt(self, inst: Instruction):
        """Square root."""
        a = self._pop_number()
        self._push(np.sqrt(a) if a >= 0 else float('nan'))
    
    def _handle_pow(self, inst: Instruction):
        """Power."""
        b = self._pop_number()
        a = self._pop_number()
        self._push(a ** b)
    
    def _handle_min(self, inst: Instruction):
        """Minimum."""
        b = self._pop_number()
        a = self._pop_number()
        self._push(min(a, b))
    
    def _handle_max(self, inst: Instruction):
        """Maximum."""
        b = self._pop_number()
        a = self._pop_number()
        self._push(max(a, b))
    
    # -------------------------------------------------------------------------
    # Opcode Handlers - Comparison Operations
    # -------------------------------------------------------------------------
    
    def _handle_eq(self, inst: Instruction):
        """Equal."""
        b = self._pop_number()
        a = self._pop_number()
        self._push(1 if abs(a - b) < 1e-10 else 0, ValueType.INT)
    
    def _handle_ne(self, inst: Instruction):
        """Not equal."""
        b = self._pop_number()
        a = self._pop_number()
        self._push(0 if abs(a - b) < 1e-10 else 1, ValueType.INT)
    
    def _handle_lt(self, inst: Instruction):
        """Less than."""
        b = self._pop_number()
        a = self._pop_number()
        self._push(1 if a < b else 0, ValueType.INT)
    
    def _handle_le(self, inst: Instruction):
        """Less than or equal."""
        b = self._pop_number()
        a = self._pop_number()
        self._push(1 if a <= b else 0, ValueType.INT)
    
    def _handle_gt(self, inst: Instruction):
        """Greater than."""
        b = self._pop_number()
        a = self._pop_number()
        self._push(1 if a > b else 0, ValueType.INT)
    
    # -------------------------------------------------------------------------
    # Opcode Handlers - Physics Operations
    # -------------------------------------------------------------------------
    
    def _handle_grad(self, inst: Instruction):
        """Compute gradient: ∇f"""
        # Use first field if available, no stack pop required
        field_name = None
        if self.stack and self._peek().value_type == ValueType.FIELD:
            field_name = self._pop().value
        
        if field_name and field_name in self.fields:
            field = self.fields[field_name]
        elif self.fields:
            # Use first field
            field_name = list(self.fields.keys())[0]
            field = self.fields[field_name]
        else:
            self._push([[0, 0, 0]], ValueType.TENSOR)
            return
        
        dx = inst.operand1 if inst.operand1 else 0.1
        dy = inst.operand2 if inst.operand2 else 0.1
        dz = inst.operand3 if inst.operand3 else 0.1
        
        grad = _compute_gradient_jit(field, dx, dy, dz)
        self._push(grad.tolist(), ValueType.TENSOR)
    
    def _handle_div_op(self, inst: Instruction):
        """Compute divergence: ∇·v"""
        # Use first field if available, no stack pop required
        field_name = None
        if self.stack and self._peek().value_type == ValueType.FIELD:
            field_name = self._pop().value
        
        if field_name and field_name in self.fields:
            field = self.fields[field_name]
        elif self.fields:
            field_name = list(self.fields.keys())[0]
            field = self.fields[field_name]
        else:
            self._push(0.0)
            return
        
        dx = inst.operand1 if inst.operand1 else 0.1
        dy = inst.operand2 if inst.operand2 else 0.1
        dz = inst.operand3 if inst.operand3 else 0.1
        
        div = _compute_divergence_jit(field, dx, dy, dz)
        self._push(float(np.mean(div)))
    
    def _handle_curl(self, inst: Instruction):
        """Compute curl: ∇ × v"""
        # Use first field if available, no stack pop required
        field_name = None
        if self.stack and self._peek().value_type == ValueType.FIELD:
            field_name = self._pop().value
        
        if field_name and field_name in self.fields:
            field = self.fields[field_name]
        elif self.fields:
            field_name = list(self.fields.keys())[0]
            field = self.fields[field_name]
        else:
            self._push([[0, 0, 0]], ValueType.TENSOR)
            return
        
        dx = inst.operand1 if inst.operand1 else 0.1
        dy = inst.operand2 if inst.operand2 else 0.1
        dz = inst.operand3 if inst.operand3 else 0.1
        
        curl = _compute_curl_jit(field, dx, dy, dz)
        self._push(curl.tolist(), ValueType.TENSOR)
    
    def _handle_laplacian(self, inst: Instruction):
        """Compute Laplacian: Δf = ∇²f"""
        # Use first field if available, no stack pop required
        field_name = None
        if self.stack and self._peek().value_type == ValueType.FIELD:
            field_name = self._pop().value
        
        if field_name and field_name in self.fields:
            field = self.fields[field_name]
        elif self.fields:
            field_name = list(self.fields.keys())[0]
            field = self.fields[field_name]
        else:
            self._push(0.0)
            return
        
        dx = inst.operand1 if inst.operand1 else 0.1
        dy = inst.operand2 if inst.operand2 else 0.1
        dz = inst.operand3 if inst.operand3 else 0.1
        
        lap = _compute_laplacian_jit(field, dx, dy, dz)
        self._push(lap.tolist(), ValueType.TENSOR)
    
    def _handle_ddt(self, inst: Instruction):
        """Time derivative: ∂/∂t"""
        field_name = self._pop().value
        self._push(0.0)
    
    def _handle_partial(self, inst: Instruction):
        """Partial derivative: ∂/∂x_i"""
        direction = inst.operand1
        field_name = self._pop().value
        self._push(0.0)
    
    def _handle_cov_deriv(self, inst: Instruction):
        """Covariant derivative: ∇^g"""
        metric = self._pop()
        field = self._pop()
        self._push(0.0)
    
    def _handle_lie_deriv(self, inst: Instruction):
        """Lie derivative: L_v"""
        vector = self._pop()
        field = self._pop()
        self._push(0.0)
    
    # -------------------------------------------------------------------------
    # Opcode Handlers - Geometry Operations
    # -------------------------------------------------------------------------
    
    def _handle_hodge(self, inst: Instruction):
        """Hodge star operator."""
        form = self._pop()
        self._push(0.0)
    
    def _handle_inner(self, inst: Instruction):
        """Interior product."""
        b = self._pop()
        a = self._pop()
        self._push(0.0)
    
    def _handle_wedge(self, inst: Instruction):
        """Wedge product."""
        b = self._pop()
        a = self._pop()
        self._push(0.0)
    
    def _handle_contract(self, inst: Instruction):
        """Tensor contraction."""
        index = inst.operand1
        b = self._pop()
        a = self._pop()
        self._push(0.0)
    
    def _handle_sym(self, inst: Instruction):
        """Symmetrize tensor."""
        tensor = self._pop()
        self._push(0.0)
    
    def _handle_antisym(self, inst: Instruction):
        """Antisymmetrize tensor."""
        tensor = self._pop()
        self._push(0.0)
    
    def _handle_trace(self, inst: Instruction):
        """Metric trace."""
        tensor = self._pop()
        self._push(0.0)
    
    def _handle_lower(self, inst: Instruction):
        """Lower indices with metric."""
        tensor = self._pop()
        metric = self._pop()
        self._push(0.0)
    
    def _handle_raise(self, inst: Instruction):
        """Raise indices with metric."""
        tensor = self._pop()
        metric = self._pop()
        self._push(0.0)
    
    # -------------------------------------------------------------------------
    # Opcode Handlers - Tensor Construction
    # -------------------------------------------------------------------------
    
    def _handle_make_sym6(self, inst: Instruction):
        """Construct sym6 tensor from components."""
        zz = self._pop_number()
        yz = self._pop_number()
        yy = self._pop_number()
        xz = self._pop_number()
        xy = self._pop_number()
        xx = self._pop_number()
        
        sym6 = np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])
        self._push(sym6.tolist(), ValueType.SYM6)
    
    def _handle_make_vec3(self, inst: Instruction):
        """Construct vec3 from components."""
        z = self._pop_number()
        y = self._pop_number()
        x = self._pop_number()
        
        vec3 = np.array([x, y, z])
        self._push(vec3.tolist(), ValueType.VEC3)
    
    def _handle_index_get(self, inst: Instruction):
        """Get tensor component."""
        indices = []
        for _ in range(inst.operand1):
            indices.append(self._pop_int())
        tensor = self._pop()
        self._push(0.0)
    
    def _handle_index_set(self, inst: Instruction):
        """Set tensor component."""
        value = self._pop_number()
        indices = []
        for _ in range(inst.operand1):
            indices.append(self._pop_int())
        tensor = self._pop()
    
    # -------------------------------------------------------------------------
    # Opcode Handlers - Ledger/EXEC Operations
    # -------------------------------------------------------------------------
    
    def _handle_emit(self, inst: Instruction):
        """Emit receipt to ledger."""
        receipt_type = inst.operand1
        data = self._pop()
        
        receipt_id = self._generate_receipt_id()
        receipt = Receipt(
            receipt_id=receipt_id,
            step_number=self.step_number,
            stage=self._get_current_stage(),
            operation=f"emit_{receipt_type}",
            input_hash=self._hash_value(data),
            output_hash="",
            timestamp=time.time(),
            metadata={'receipt_type': receipt_type}
        )
        
        self.receipts.append(receipt)
        self.receipt_counter += 1
        self._push(receipt_id, ValueType.STRING)
    
    def _handle_check_gate(self, inst: Instruction):
        """Check gate condition."""
        gate_id = inst.operand1
        value = self._pop_number()
        
        if gate_id in self.gates:
            threshold = self.gates[gate_id]
            if value >= threshold:
                raise GateViolationError(gate_id, value, threshold, self._ip)
        
        self._push(value)
    
    def _handle_check_inv(self, inst: Instruction):
        """Check invariant condition."""
        inv_id = inst.operand1
        value = self._pop_number()
        
        inv_key = str(inv_id)
        if inv_key in self.invariants:
            expected, tolerance = self.invariants[inv_key]
            if abs(value - expected) > tolerance:
                raise InvariantViolationError(inv_key, value, expected, self._ip)
        
        self._push(value)
    
    def _handle_check_invariant(self, inst: Instruction):
        """Check invariant condition (full name)."""
        inv_id = inst.immediate
        value = self._pop_number()
        
        inv_key = str(inv_id)
        if inv_key in self.invariants:
            expected, tolerance = self.invariants[inv_key]
            if abs(value - expected) > tolerance:
                raise InvariantViolationError(inv_key, value, expected, self._ip)
        
        self._push(value)
    
    def _handle_seal(self, inst: Instruction):
        """Seal step (commit to ledger)."""
        state_hash = self._hash_state()
        
        for receipt in self.receipts:
            if not receipt.output_hash:
                receipt.output_hash = state_hash
        
        self._push(1 if state_hash else 0, ValueType.INT)
    
    def _handle_verify(self, inst: Instruction):
        """Verify receipt."""
        receipt_id = self._pop().value
        self._push(1, ValueType.INT)
    
    def _handle_get_receipt(self, inst: Instruction):
        """Get receipt by ID."""
        receipt_id = self._pop().value
        for r in self.receipts:
            if r.receipt_id == receipt_id:
                self._push(str(r), ValueType.STRING)
                return
        self._push("", ValueType.STRING)
    
    # -------------------------------------------------------------------------
    # Opcode Handlers - Constraint Operations
    # -------------------------------------------------------------------------
    
    def _handle_enforce_h(self, inst: Instruction):
        """Enforce Hamiltonian constraint."""
        self._push(0.0)
    
    def _handle_enforce_m(self, inst: Instruction):
        """Enforce momentum constraint."""
        self._push(0.0)
    
    def _handle_project(self, inst: Instruction):
        """Project to constraint surface."""
        self._push(0.0)
    
    def _handle_phi_func(self, inst: Instruction):
        """PHI function for constraints."""
        self._push(0.0)
    
    # -------------------------------------------------------------------------
    # Opcode Handlers - Stage Control
    # -------------------------------------------------------------------------
    
    def _handle_stage_enter(self, inst: Instruction):
        """Enter new stage."""
        stage_name = inst.immediate
        self._push(stage_name, ValueType.STAGE)
    
    def _handle_stage_exit(self, inst: Instruction):
        """Exit current stage."""
        pass
    
    def _handle_stage_sync(self, inst: Instruction):
        """Synchronize across stages."""
        pass
    
    def _handle_ckpt(self, inst: Instruction):
        """Create checkpoint."""
        checkpoint = self.snapshot()
        self._push(checkpoint, ValueType.TENSOR)
    
    def _handle_restore(self, inst: Instruction):
        """Restore from checkpoint."""
        checkpoint = self._pop()
    
    # -------------------------------------------------------------------------
    # Opcode Handlers - JIT/Markers
    # -------------------------------------------------------------------------
    
    def _handle_jit_start(self, inst: Instruction):
        """Mark JIT compilation start."""
        pass
    
    def _handle_jit_end(self, inst: Instruction):
        """Mark JIT compilation end."""
        pass
    
    def _handle_profile(self, inst: Instruction):
        """Profile counter."""
        counter = inst.operand1
        self._push(self.step_number, ValueType.INT)
    
    # -------------------------------------------------------------------------
    # Opcode Handlers - GR Operations
    # -------------------------------------------------------------------------
    
    def _handle_christoffel(self, inst: Instruction):
        """Compute Christoffel symbols."""
        metric = self._pop()
        self._push([[[0]]], ValueType.TENSOR)
    
    def _handle_ricci(self, inst: Instruction):
        """Compute Ricci tensor."""
        metric = self._pop()
        self._push([[[0]]], ValueType.TENSOR)
    
    def _handle_ricci_scalar(self, inst: Instruction):
        """Compute Ricci scalar."""
        metric = self._pop()
        self._push(0.0)
    
    def _handle_einstein(self, inst: Instruction):
        """Compute Einstein tensor."""
        metric = self._pop()
        self._push([[[0]]], ValueType.TENSOR)
    
    def _handle_riemann(self, inst: Instruction):
        """Compute Riemann tensor."""
        metric = self._pop()
        self._push([[[[0]]]], ValueType.TENSOR)
    
    # -------------------------------------------------------------------------
    # Opcode Handlers - BSSN Operations
    # -------------------------------------------------------------------------
    
    def _handle_bssn_conform(self, inst: Instruction):
        """BSSN conformal transformation."""
        self._push(0.0)
    
    def _handle_bssn_trace(self, inst: Instruction):
        """BSSN trace of extrinsic curvature."""
        self._push(0.0)
    
    def _handle_bssn_gamma(self, inst: Instruction):
        """BSSN gamma driver."""
        self._push(0.0)
    
    def _handle_bssn_lambda(self, inst: Instruction):
        """BSSN lambda driver."""
        self._push(0.0)
    
    # -------------------------------------------------------------------------
    # Opcode Handlers - Gauge Operations
    # -------------------------------------------------------------------------
    
    def _handle_gauge_lapse(self, inst: Instruction):
        """Gauge fixing for lapse."""
        self._push(0.0)
    
    def _handle_gauge_shift(self, inst: Instruction):
        """Gauge fixing for shift."""
        self._push(0.0)
    
    def _handle_gauge_singularity(self, inst: Instruction):
        """Singularity handling."""
        pass
    
    # -------------------------------------------------------------------------
    # JIT Compilation
    # -------------------------------------------------------------------------
    
    def _execute_jit(self, ip: int):
        """Execute JIT-compiled code."""
        if ip in self.jit_cache:
            self.jit_cache[ip]()
    
    def try_jit_compile(self, start_ip: int, end_ip: int):
        """Try to JIT compile a code segment."""
        if self.jit_tier >= 2:
            return
        self.jit_tier = min(self.jit_tier + 1, 2)
    
    # -------------------------------------------------------------------------
    # Helper Methods
    # -------------------------------------------------------------------------
    
    def _get_result(self) -> Any:
        """Get final result from stack."""
        if self.stack:
            return self.stack[-1].value
        return None
    
    def _get_current_stage(self) -> str:
        """Get current execution stage."""
        stages = ["OBSERVE", "DECIDE", "ACT_PHY", "ACT_CONS", "AUDIT", "ACCEPT"]
        stage_idx = self.step_number % len(stages)
        return stages[stage_idx]
    
    def _generate_receipt_id(self) -> str:
        """Generate unique receipt ID."""
        timestamp = str(time.time())
        counter = str(self.receipt_counter)
        hash_input = f"{timestamp}{counter}{self.step_number}"
        return hashlib.sha256(hash_input.encode()).hexdigest()[:16]
    
    def _hash_value(self, value: RuntimeValue) -> str:
        """Hash a runtime value."""
        data = str(value.value).encode()
        return hashlib.sha256(data).hexdigest()[:16]
    
    def _hash_state(self) -> str:
        """Hash current VM state."""
        state_data = {
            'step': self.step_number,
            'stack_depth': len(self.stack),
            'ip': self._ip,
            'registers': [r.value for r in self.registers[:8]],
        }
        data = str(state_data).encode()
        return hashlib.sha256(data).hexdigest()[:16]
    
    def snapshot(self) -> bytes:
        """Create checkpoint snapshot."""
        state = {
            'ip': self._ip,
            'sp': self._sp,
            'step': self.step_number,
            'stack': [rv.value for rv in self.stack],
            'registers': [rv.value for rv in self.registers],
        }
        return str(state).encode()
    
    def restore(self, data: bytes):
        """Restore from checkpoint."""
        state = eval(data.decode())
        self._ip = state['ip']
        self._sp = state['sp']
        self.step_number = state['step']
        self.stack = [RuntimeValue(v, ValueType.FLOAT) for v in state['stack']]
        self.registers = [RuntimeValue(v, ValueType.FLOAT) for v in state['registers']]
    
    # -------------------------------------------------------------------------
    # Determinism Verification
    # -------------------------------------------------------------------------
    
    def verify_determinism(self) -> Tuple[bool, float]:
        """Verify execution is deterministic."""
        if len(self.operation_log) < 2:
            return True, 0.0
        
        base_ops = [op['opcode'] for op in self.operation_log[0]['operations']]
        max_deviation = 0.0
        
        for log in self.operation_log[1:]:
            ops = [op['opcode'] for op in log['operations']]
            if len(ops) != len(base_ops):
                return False, 1.0
            
            differences = sum(1 for a, b in zip(base_ops, ops) if a != b)
            deviation = differences / len(base_ops)
            max_deviation = max(max_deviation, deviation)
        
        return max_deviation < 1e-10, max_deviation
    
    def get_operation_log(self) -> List[Dict]:
        """Get operation log for determinism verification."""
        return self.operation_log
    
    def get_execution_stats(self) -> Dict:
        """Get execution statistics."""
        return {
            'step_count': self.step_number,
            'instructions_executed': len(self.operation_log),
            'receipts_issued': len(self.receipts),
            'execution_time': self.clocks.get('execution_time', 0),
            'state': self.state,
            'stack_depth': len(self.stack),
            'jit_tier': self.jit_tier,
        }
