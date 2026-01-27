"""
Mem2Reg Optimization Pass for NIR (NLL Compiler)

This pass promotes stack-allocated variables to SSA registers where possible.
Variables that are only assigned once (and never have their address taken)
can be converted from stack allocations to direct SSA values.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any
from src.nllc.nir import (
    Module, Function, BasicBlock, Instruction,
    AllocInst, StoreInst, LoadInst, ConstInst, BinOpInst,
    CallInst, IntrinsicCallInst, GetElementInst, BrInst, RetInst,
    Value, Type, Trace
)


@dataclass
class Definition:
    """Represents a single definition of a variable."""
    value: Value
    store_inst: Optional[StoreInst] = None
    block_name: str = ""


@dataclass
class VariableInfo:
    """Information about a stack-allocated variable."""
    alloc: AllocInst
    definitions: List[Definition] = field(default_factory=list)
    uses: List[tuple] = field(default_factory=list)  # (LoadInst, block_name)
    is_address_taken: bool = False
    is_single_definition: bool = False


def mem2reg_optimize(module: Module, enable: bool = True) -> Module:
    """
    Apply Mem2Reg optimization to the module.
    
    Args:
        module: The NIR Module to optimize
        enable: Whether to enable the optimization (default: True)
    
    Returns:
        Optimized Module with stack variables promoted to SSA registers
    """
    if not enable:
        return module
    
    # Process each function
    for func in module.functions:
        _optimize_function(func)
    
    return module


def _optimize_function(func: Function):
    """Apply Mem2Reg to a single function."""
    
    # Step 1: Collect all stack-allocated variables and their uses
    var_infos: Dict[str, VariableInfo] = {}
    
    # First pass: find allocs and classify variables
    for block in func.blocks:
        for inst in block.instructions:
            if isinstance(inst, AllocInst):
                var_name = inst.result.name
                var_infos[var_name] = VariableInfo(alloc=inst)
            elif isinstance(inst, StoreInst):
                # Track if address is taken
                ptr_name = inst.ptr.name
                if ptr_name in var_infos:
                    var_infos[ptr_name].definitions.append(
                        Definition(value=inst.value, store_inst=inst, block_name=block.name)
                    )
            elif isinstance(inst, LoadInst):
                ptr_name = inst.ptr.name
                if ptr_name in var_infos:
                    var_infos[ptr_name].uses.append((inst, block.name))
                    # LoadInst is NOT address-taking - it's just reading the value
                    # We only mark address-taken for GetElementInst or when the ptr
                    # is passed to functions (we'll add that later)
            elif isinstance(inst, GetElementInst):
                # Array indexing means address is taken
                array_name = inst.array.name
                if array_name in var_infos:
                    var_infos[array_name].is_address_taken = True
    
    # Step 2: Identify promotable variables (single definition, not address taken)
    promotable_vars: Dict[str, Value] = {}  # var_name -> replacement value
    
    for var_name, info in var_infos.items():
        info.is_single_definition = len(info.definitions) == 1
        
        # A variable is promotable if:
        # 1. Only one definition
        # 2. Address is not taken (no loads, no GetElementInst)
        # 3. The definition value is known (ConstInst or computed value)
        if info.is_single_definition and not info.is_address_taken:
            definition = info.definitions[0]
            if definition.value:
                promotable_vars[var_name] = definition.value
    
    # Step 3: Build a map of all values that might need replacement
    # Map: original value name -> replacement value
    value_mapping: Dict[str, Value] = {}
    
    for var_name, replacement in promotable_vars.items():
        value_mapping[var_name] = replacement
    
    # Step 4: Process instructions and replace uses
    new_instructions: List[Instruction] = []
    alloc_to_remove: Set[str] = set()
    store_to_remove: Set[int] = set()  # by instruction id
    load_to_remove: Set[int] = set()
    
    for block in func.blocks:
        new_block_instrs = []
        for inst in block.instructions:
            inst_id = id(inst)
            
            if isinstance(inst, AllocInst):
                if inst.result.name in promotable_vars:
                    alloc_to_remove.add(inst.result.name)
                    continue  # Skip this alloc
                new_block_instrs.append(inst)
                
            elif isinstance(inst, StoreInst):
                if inst.ptr.name in promotable_vars:
                    store_to_remove.add(inst_id)
                    continue  # Skip this store
                new_block_instrs.append(inst)
                
            elif isinstance(inst, LoadInst):
                if inst.ptr.name in promotable_vars:
                    # Replace the load result with the stored value
                    # IMPORTANT: Map BEFORE skipping, so subsequent instructions can use it
                    replacement = value_mapping.get(inst.ptr.name)
                    if replacement:
                        # Track that uses of the load's result should use the replacement
                        value_mapping[inst.result.name] = replacement
                    load_to_remove.add(inst_id)
                    continue  # Skip this load
                new_block_instrs.append(inst)
                
            elif isinstance(inst, ConstInst):
                # Track const values
                value_mapping[inst.result.name] = inst.result
                new_block_instrs.append(inst)
                
            elif isinstance(inst, BinOpInst):
                # Replace operands that are mapped
                new_left = value_mapping.get(inst.left.name, inst.left)
                new_right = value_mapping.get(inst.right.name, inst.right)
                inst.left = new_left
                inst.right = new_right
                # Track the result value
                value_mapping[inst.result.name] = inst.result
                new_block_instrs.append(inst)
                
            elif isinstance(inst, CallInst):
                # Replace args that are mapped
                new_args = [value_mapping.get(arg.name, arg) for arg in inst.args]
                inst.args = new_args
                if inst.result:
                    value_mapping[inst.result.name] = inst.result
                new_block_instrs.append(inst)
                
            elif isinstance(inst, IntrinsicCallInst):
                # Replace args that are mapped
                new_args = [value_mapping.get(arg.name, arg) for arg in inst.args]
                inst.args = new_args
                if inst.result:
                    value_mapping[inst.result.name] = inst.result
                new_block_instrs.append(inst)
                
            elif isinstance(inst, GetElementInst):
                # Replace array that is mapped
                new_array = value_mapping.get(inst.array.name, inst.array)
                inst.array = new_array
                value_mapping[inst.result.name] = inst.result
                new_block_instrs.append(inst)
                
            elif isinstance(inst, BrInst):
                # Replace cond if mapped
                if inst.cond:
                    new_cond = value_mapping.get(inst.cond.name, inst.cond)
                    inst.cond = new_cond
                new_block_instrs.append(inst)
                
            elif isinstance(inst, RetInst):
                # Replace value if mapped
                if inst.value:
                    new_val = value_mapping.get(inst.value.name, inst.value)
                    inst.value = new_val
                new_block_instrs.append(inst)
                
            else:
                new_block_instrs.append(inst)
        
        block.instructions = new_block_instrs
    
    # Step 5: Handle phi-like constructs (not needed for single-definition)
    # Since we only promote single-definition variables, no phi nodes needed


def mem2reg_transform_example():
    """
    Example showing the Mem2Reg transformation.
    
    Before (stack-allocated):
        %0 = alloc Float        # AllocInst
        store %0, 1.0           # StoreInst
        %1 = load %0            # LoadInst
        %2 = add %1, 2.0        # BinOpInst
    
    After (SSA register):
        %0 = 1.0                # ConstInst
        %1 = add %0, 2.0        # BinOpInst (no load/store)
    """
    # Create a simple module for demonstration
    from src.nllc.nir import (
        IntType, FloatType, Trace, Module, Function, BasicBlock,
        Value, AllocInst, StoreInst, LoadInst, ConstInst, BinOpInst, RetInst
    )
    from src.nllc.ast import Span
    
    span = Span(start=0, end=100)
    trace = Trace("example.nllc", span, "example")
    
    # Create a simple function with stack-allocated variable
    func = Function("example_func", [], FloatType(), [])
    
    entry = BasicBlock("entry")
    func.blocks.append(entry)
    
    # %0 = alloc Float
    alloc_result = Value("%0", FloatType())
    alloc_inst = AllocInst(trace, alloc_result, FloatType())
    entry.instructions.append(alloc_inst)
    
    # store %0, 1.0
    const_val = Value("%const_1_0", FloatType())
    const_inst = ConstInst(trace, const_val, 1.0)
    entry.instructions.append(const_inst)
    
    store_inst = StoreInst(trace, alloc_result, None, const_val)
    entry.instructions.append(store_inst)
    
    # %1 = load %0
    load_result = Value("%1", FloatType())
    load_inst = LoadInst(trace, load_result, alloc_result)
    entry.instructions.append(load_inst)
    
    # %2 = add %1, 2.0
    const_val_2 = Value("%const_2_0", FloatType())
    const_inst_2 = ConstInst(trace, const_val_2, 2.0)
    entry.instructions.append(const_inst_2)
    
    binop_result = Value("%2", FloatType())
    binop_inst = BinOpInst(trace, binop_result, load_result, "+", const_val_2)
    entry.instructions.append(binop_inst)
    
    # return %2
    ret_inst = RetInst(trace, binop_result)
    entry.instructions.append(ret_inst)
    
    module = Module([func])
    
    print("Before Mem2Reg:")
    _print_module(module)
    
    # Apply Mem2Reg
    optimized = mem2reg_optimize(module)
    
    print("\nAfter Mem2Reg:")
    _print_module(optimized)
    
    return optimized


def _print_module(module: Module):
    """Print a module in a readable format."""
    for func in module.functions:
        print(f"func @{func.name}() -> {func.return_ty.__class__.__name__}")
        for block in func.blocks:
            print(f"  {block.name}:")
            for inst in block.instructions:
                print(f"    {inst}")


# Conditional export for integration with nir.py
if __name__ == "__main__":
    # Demo the transformation
    mem2reg_transform_example()
