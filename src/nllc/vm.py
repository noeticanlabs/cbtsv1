import hashlib
import json
import dataclasses
from typing import Dict, List, Any, Optional
from src.nllc.nir import *
from src.nllc.intrinsic_binder import IntrinsicBinder
from src.nllc.type_checker import TypeChecker, typecheck_nir_module
from src.common.receipt import create_run_receipt
from src.core.gr_core_fields import inv_sym6, trace_sym6, sym6_to_mat33, mat33_to_sym6, det_sym6, norm2_sym6

class VM:
    def __init__(self, module: Module, module_id: str, dep_closure_hash: str, gr_host_api=None, typecheck: bool = True):
        self.module = module
        self.module_id = module_id
        self.dep_closure_hash = dep_closure_hash
        self.functions: Dict[str, Function] = {f.name: f for f in module.functions}
        self.call_stack: List[Dict[str, Any]] = []
        self.receipts: List[dict] = []  # for summary, but also step receipts
        self.step_counter = 0
        self.state_snapshots: List[Dict[str, Any]] = []  # For rollback
        self.gr_host_api = gr_host_api  # GR integration
        self.binder = IntrinsicBinder()
        self.typecheck_enabled = typecheck
        
        # Run type checking if enabled
        if self.typecheck_enabled:
            self._run_type_check()
    
    def _run_type_check(self) -> None:
        """Run type checking on the module before execution."""
        success, errors = typecheck_nir_module(self.module, raise_on_error=False)
        if not success:
            error_messages = "\n".join(str(e) for e in errors)
            raise TypeError(f"Type checking failed:\n{error_messages}")

    def run(self) -> Any:
        # Start with main function
        if "main" not in self.functions:
            raise ValueError("No main function")
        main_func = self.functions["main"]
        result = self.call_function(main_func, [])
        # Emit summary receipt
        summary_receipt = self.create_summary_receipt(result)
        self.receipts.append(summary_receipt)
        return result

    def call_function(self, func: Function, args: List[Any]) -> Any:
        # Create frame
        frame = {
            'func': func,
            'env': {},
            'block_idx': 0,
            'inst_idx': 0,
            'return_value': None
        }
        # Bind parameters
        for param, arg in zip(func.params, args):
            frame['env'][param.name] = arg
        self.call_stack.append(frame)

        while True:
            block = func.blocks[frame['block_idx']]
            if frame['inst_idx'] >= len(block.instructions):
                # End of block, should have terminator
                raise ValueError("Block without terminator")
            inst = block.instructions[frame['inst_idx']]

            # Emit step receipt for instruction
            step_receipt = self.create_step_receipt(inst, f"block_{block.name}_inst_{frame['inst_idx']}")
            self.receipts.append(step_receipt)

            if isinstance(inst, RetInst):
                self.call_stack.pop()
                return inst.value and frame['env'].get(inst.value.name) if inst.value else None
            elif isinstance(inst, BrInst):
                if inst.cond is None:
                    # Unconditional
                    next_block = inst.true_block
                else:
                    cond_val = frame['env'][inst.cond.name]
                    next_block = inst.true_block if cond_val else inst.false_block
                # Find block index
                for i, b in enumerate(func.blocks):
                    if b.name == next_block:
                        frame['block_idx'] = i
                        frame['inst_idx'] = 0
                        break
                else:
                    raise ValueError(f"Block {next_block} not found")
            else:
                # Execute instruction
                self.execute_instruction(inst, frame['env'])
                frame['inst_idx'] += 1

    def execute_instruction(self, inst: Instruction, env: Dict[str, Any]):
        if isinstance(inst, ConstInst):
            if isinstance(inst.value, list):
                env[inst.result.name] = [env[name] for name in inst.value]
            elif isinstance(inst.value, dict):
                env[inst.result.name] = {k: env[v] for k, v in inst.value.items()}
            else:
                env[inst.result.name] = inst.value
        elif isinstance(inst, BinOpInst):
            try:
                left = env[inst.left.name]
                right = env[inst.right.name]
            except KeyError as e:
                print(f"KeyError in BinOpInst: {e}, inst: {inst}, env keys: {list(env.keys())}")
                raise
            if inst.op == '+':
                result = left + right
            elif inst.op == '-':
                result = left - right
            elif inst.op == '*':
                result = left * right
            elif inst.op == '/':
                result = left / right if isinstance(left, float) or isinstance(right, float) else left // right
            elif inst.op == '==':
                result = left == right
            elif inst.op == '!=':
                result = left != right
            elif inst.op == '<':
                result = left < right
            elif inst.op == '>':
                result = left > right
            elif inst.op == '<=':
                result = left <= right
            elif inst.op == '>=':
                result = left >= right
            elif inst.op == 'and':
                result = left and right
            elif inst.op == 'or':
                result = left or right
            else:
                raise NotImplementedError(f"Op {inst.op}")
            env[inst.result.name] = result
        elif isinstance(inst, CallInst):
            func_name = inst.func
            if func_name not in self.functions:
                # Built-in?
                args = [env[arg.name] for arg in inst.args]
                result = self.call_builtin(func_name, args)
            else:
                func = self.functions[func_name]
                args = [env[arg.name] for arg in inst.args]
                result = self.call_function(func, args)
            if inst.result:
                env[inst.result.name] = result
        elif isinstance(inst, IntrinsicCallInst):
            func_name = inst.func
            if self.binder.is_bound(func_name):
                kernel = self.binder.get_kernel(func_name)
                args = [env[arg.name] for arg in inst.args]
                result = kernel(*args)
            else:
                # Fallback to builtin or error
                args = [env[arg.name] for arg in inst.args]
                result = self.call_builtin(func_name, args)
            if inst.result:
                env[inst.result.name] = result
        elif isinstance(inst, LoadInst):
            # For now, assume variables are direct
            env[inst.result.name] = env[inst.ptr.name]
        elif isinstance(inst, StoreInst):
            if inst.index:
                # array[index] = value
                array = env[inst.ptr.name]
                index = env[inst.index.name]
                value = env[inst.value.name]
                array[index] = value
            else:
                # var = value
                env[inst.ptr.name] = env[inst.value.name]
        elif isinstance(inst, AllocInst):
            if isinstance(inst.ty, ArrayType):
                env[inst.result.name] = []
            elif isinstance(inst.ty, ObjectType):
                env[inst.result.name] = {}
            else:
                # Scalars (Int, Float, Bool, Str) initialized to None (or default)
                env[inst.result.name] = None
        elif isinstance(inst, GetElementInst):
            array = env[inst.array.name]
            index = env[inst.index.name]
            env[inst.result.name] = array[index]
        # No wall-clock times, deterministic ops

    def call_builtin(self, name: str, args: List[Any]) -> Any:
        # Implement built-ins deterministically
        if name == 'print':
            print(*args)
            return None
        elif name == 'len':
            return len(args[0])
        elif name == 'str':
            return str(args[0]) if args else ""
        elif name == 'abs':
            return abs(args[0]) if args else 0
        elif name == 'inv_sym6':
            return inv_sym6(*args)
        elif name == 'trace_sym6':
            return trace_sym6(*args)
        elif name == 'sym6_to_mat33':
            return sym6_to_mat33(*args)
        elif name == 'mat33_to_sym6':
            return mat33_to_sym6(*args)
        elif name == 'det_sym6':
            return det_sym6(*args)
        elif name == 'norm2_sym6':
            return norm2_sym6(*args)
        # GR Integration built-ins - try dynamic dispatch first
        if hasattr(self.gr_host_api, name):
            return getattr(self.gr_host_api, name)(*args)
        # Fallback to hardcoded if needed
        raise NotImplementedError(f"Builtin {name}")

    def create_step_receipt(self, inst: Instruction, step_id: str) -> dict:
        # Trace digest from instruction trace
        trace_str = f"{inst.trace.file}:{inst.trace.span.start}-{inst.trace.span.end}:{inst.trace.ast_path}"
        trace_digest = hashlib.sha256(trace_str.encode()).hexdigest()
        prev_id = self.receipts[-1]['receipt']['id'] if self.receipts else None
        receipt = create_run_receipt(step_id, trace_digest, prev_id)
        return {
            'type': 'step',
            'receipt': dataclasses.asdict(receipt),
            'module_id': self.module_id,
            'dep_closure_hash': self.dep_closure_hash
        }

    def create_summary_receipt(self, result: Any) -> dict:
        # Summary receipt
        summary_data = f"execution_result:{result}"
        summary_digest = hashlib.sha256(summary_data.encode()).hexdigest()
        prev_id = self.receipts[-1]['receipt']['id'] if self.receipts else None
        receipt = create_run_receipt("summary", summary_digest, prev_id)
        return {
            'type': 'summary',
            'receipt': dataclasses.asdict(receipt),
            'module_id': self.module_id,
            'dep_closure_hash': self.dep_closure_hash,
            'result': result
        }

    def get_receipts(self) -> List[dict]:
        return self.receipts

    def get_receipts_json(self) -> str:
        # Canonical JSON
        return json.dumps(self.receipts, sort_keys=True, separators=(',', ':'))

    def snapshot_state(self):
        """Save current env state for rollback."""
        if self.call_stack:
            self.state_snapshots.append(self.call_stack[-1]['env'].copy())

    def rollback_state(self):
        """Restore last snapshot."""
        if self.state_snapshots:
            self.call_stack[-1]['env'] = self.state_snapshots.pop()

    def execute_block(self, block: BasicBlock, env: Dict[str, Any]) -> Any:
        """Execute a single basic block with given env, return last instruction result."""
        inst_idx = 0
        while inst_idx < len(block.instructions):
            inst = block.instructions[inst_idx]
            self.execute_instruction(inst, env)
            inst_idx += 1
            if isinstance(inst, (BrInst, RetInst)):
                # For simplicity, assume blocks don't branch out
                break
        # Return value of last store or something, for audit assume last inst is const or binop for bool
        return env.get(inst.result.name) if hasattr(inst, 'result') else None