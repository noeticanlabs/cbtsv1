"""
Dead Code Elimination (DCE) Pass for NIR

This pass removes dead code from NIR modules:
1. Instructions whose results are never used
2. Unreachable basic blocks
3. Functions that are never called (whole-program DCE)

Usage:
    from src.nllc.dead_code_elimination import DeadCodeElimination
    
    dce = DeadCodeElimination()
    module = dce.remove_dead_code(module)
"""

from dataclasses import dataclass, field
from typing import Set, List, Dict, Optional, Tuple
from nllc.nir import (
    Module, Function, BasicBlock, Instruction,
    ConstInst, BinOpInst, CallInst, IntrinsicCallInst,
    LoadInst, StoreInst, AllocInst, GetElementInst,
    BrInst, RetInst, Value
)


@dataclass
class DCEStats:
    """Statistics from dead code elimination."""
    instructions_removed: int = 0
    blocks_removed: int = 0
    values_removed: int = 0
    
    def __str__(self) -> str:
        return (f"DCE: removed {self.instructions_removed} instructions, "
                f"{self.blocks_removed} blocks, {self.values_removed} values")


class DeadCodeElimination:
    """
    Dead Code Elimination pass for NIR.
    
    This pass performs two types of dead code elimination:
    1. Local DCE: Removes instructions whose results are never used within
       the same basic block.
    2. Global DCE: Removes unreachable basic blocks and functions that
       are never called.
    3. Whole-program DCE: Removes functions that are not reachable from main.
    
    The pass uses a liveness analysis to determine which values are live
    (used by other instructions) and removes the rest.
    """
    
    def __init__(self):
        self.stats = DCEStats()
    
    def remove_dead_code(self, module: Module) -> Tuple[Module, DCEStats]:
        """
        Remove all dead code from a module.
        
        Args:
            module: The NIR module to optimize.
            
        Returns:
            Tuple of (optimized module, statistics).
        """
        self.stats = DCEStats()
        
        # Global DCE: Remove unreachable functions
        module = self._remove_unreachable_functions(module)
        
        # For each function, perform local and global DCE
        for func in module.functions[:]:  # Copy list to allow modification
            optimized_func, func_stats = self._optimize_function(func)
            self.stats.instructions_removed += func_stats.instructions_removed
            self.stats.blocks_removed += func_stats.blocks_removed
            
            # Update function in place
            idx = module.functions.index(func)
            module.functions[idx] = optimized_func
        
        return module, self.stats
    
    def _remove_unreachable_functions(self, module: Module) -> Module:
        """Remove functions that are never called (whole-program DCE)."""
        # Find all called functions
        called_funcs: Set[str] = set()
        
        for func in module.functions:
            for block in func.blocks:
                for inst in block.instructions:
                    if isinstance(inst, (CallInst, IntrinsicCallInst)):
                        if inst.func not in ('print', 'len', 'str', 'abs', 
                                            'inv_sym6', 'trace_sym6', 'sym6_to_mat33',
                                            'mat33_to_sym6', 'det_sym6', 'norm2_sym6'):
                            called_funcs.add(inst.func)
        
        # Always keep main
        called_funcs.add('main')
        
        # Remove unreachable functions
        reachable_funcs = []
        for func in module.functions:
            if func.name in called_funcs:
                reachable_funcs.append(func)
            else:
                self.stats.values_removed += 1
        
        module.functions = reachable_funcs
        return module
    
    def _optimize_function(self, func: Function) -> Tuple[Function, DCEStats]:
        """Optimize a single function."""
        stats = DCEStats()
        
        # Build liveness information
        live_values = self._compute_live_values(func)
        
        # Remove dead instructions from each block
        for block in func.blocks:
            original_len = len(block.instructions)
            block.instructions = [inst for inst in block.instructions 
                                  if self._is_inst_live(inst, live_values)]
            stats.instructions_removed += original_len - len(block.instructions)
        
        # Remove unreachable blocks
        reachable_blocks = self._find_reachable_blocks(func)
        original_block_count = len(func.blocks)
        func.blocks = [block for block in func.blocks if block.name in reachable_blocks]
        stats.blocks_removed += original_block_count - len(func.blocks)
        
        return func, stats
    
    def _compute_live_values(self, func: Function) -> Dict[str, Set[str]]:
        """
        Compute liveness information for a function.
        
        Returns a dict mapping block names to sets of live values.
        A value is live if it is used by a later instruction or by
        a successor block.
        """
        # Initialize: all values are potentially dead
        live_in: Dict[str, Set[str]] = {b.name: set() for b in func.blocks}
        live_out: Dict[str, Set[str]] = {b.name: set() for b in func.blocks}
        
        # For each block, collect:
        # - defs: values defined in this block
        # - uses: values used in this block (before their definition)
        block_defs: Dict[str, Set[str]] = {b.name: set() for b in func.blocks}
        block_uses: Dict[str, Set[str]] = {b.name: set() for b in func.blocks}
        
        for block in func.blocks:
            for inst in block.instructions:
                # Record uses (values read)
                for arg_name in self._get_inst_args(inst):
                    if arg_name not in block_defs.get(block.name, set()):
                        block_uses[block.name].add(arg_name)
                
                # Record defs (values written)
                result = getattr(inst, 'result', None) or getattr(inst, 'value', None)
                if result and isinstance(result, Value):
                    block_defs[block.name].add(result.name)
        
        # Iterative dataflow analysis
        changed = True
        while changed:
            changed = False
            for block in func.blocks:
                # live_out[b] = union of live_in[successors]
                succ_live = set()
                for inst in block.instructions:
                    if isinstance(inst, BrInst) and inst.true_block:
                        succ_live.update(live_in.get(inst.true_block, set()))
                    if isinstance(inst, BrInst) and inst.false_block:
                        succ_live.update(live_in.get(inst.false_block, set()))
                
                # live_in[b] = uses[b] union (live_out[b] minus defs[b])
                new_live_in = block_uses[block.name].union(
                    succ_live - block_defs[block.name]
                )
                
                if live_in[block.name] != new_live_in:
                    live_in[block.name] = new_live_in
                    changed = True
        
        return live_in
    
    def _get_inst_args(self, inst: Instruction) -> List[str]:
        """Get all value names used by an instruction."""
        args = []
        if hasattr(inst, 'left') and inst.left:
            args.append(inst.left.name)
        if hasattr(inst, 'right') and inst.right:
            args.append(inst.right.name)
        if hasattr(inst, 'args'):
            args.extend(arg.name for arg in inst.args if hasattr(arg, 'name'))
        if hasattr(inst, 'ptr') and inst.ptr:
            args.append(inst.ptr.name)
        if hasattr(inst, 'index') and inst.index:
            args.append(inst.index.name)
        if hasattr(inst, 'value') and inst.value:
            # Value can be an int/float/bool (not a Value object)
            if isinstance(inst.value, Value):
                args.append(inst.value.name)
        if hasattr(inst, 'cond') and inst.cond:
            args.append(inst.cond.name)
        return args
    
    def _is_inst_live(self, inst: Instruction, live_values: Dict[str, Set[str]]) -> bool:
        """Check if an instruction is live (its result is used)."""
        # Instructions that always have side effects are always live
        if isinstance(inst, (CallInst, IntrinsicCallInst, StoreInst, BrInst)):
            return True
        
        # RetInst is live (it's a terminator)
        if isinstance(inst, RetInst):
            return True
        
        # Check if result is used (use 'result' or 'value' depending on instruction)
        result = getattr(inst, 'result', None) or getattr(inst, 'value', None)
        if result:
            if isinstance(result, Value):
                block_live = live_values.get(result.name, set())
                return result.name in block_live
        
        return True
    
    def _find_reachable_blocks(self, func: Function) -> Set[str]:
        """Find all blocks reachable from entry."""
        reachable: Set[str] = set()
        stack: List[str] = ['entry']  # Assume entry block exists
        
        while stack:
            block_name = stack.pop()
            if block_name in reachable:
                continue
            reachable.add(block_name)
            
            # Find the block
            for block in func.blocks:
                if block.name == block_name:
                    # Add successors
                    for inst in block.instructions:
                        if isinstance(inst, BrInst):
                            if inst.true_block and inst.true_block not in reachable:
                                stack.append(inst.true_block)
                            if inst.false_block and inst.false_block not in reachable:
                                stack.append(inst.false_block)
                    break
        
        return reachable


def remove_dead_code(module: Module) -> Tuple[Module, DCEStats]:
    """
    Convenience function to remove dead code from a module.
    
    Args:
        module: The NIR module to optimize.
        
    Returns:
        Tuple of (optimized module, statistics).
    """
    dce = DeadCodeElimination()
    return dce.remove_dead_code(module)
