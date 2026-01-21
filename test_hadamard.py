#!/usr/bin/env python3
"""
Test Hadamard Glyph System
"""

import numpy as np
import time
from src.hadamard import HadamardAssembler, HadamardVM, HadamardCompiler
from src.solver.pir import PIRProgram, Operator, Field, Boundary, Integrator, StepLoop

def test_hadamard_basic():
    # Create assembler
    assembler = HadamardAssembler()
    assembler.add_instruction('φ', arg1=0, arg2=0)  # Load field 0 to reg 0
    assembler.add_instruction('∇²', arg1=1, arg2=0)  # Laplacian of reg 0 to reg 1
    assembler.add_instruction('⊕', arg1=2, arg2=1, meta=1)  # reg2 = reg1 + 1 * reg2 (placeholder)
    assembler.add_instruction('=', arg1=1, arg2=2)  # Assign reg2 to field 1
    bytecode = assembler.get_bytecode()

    # Create VM with dummy fields
    fields = {'field0': np.random.rand(10,10,10), 'field1': np.zeros((10,10,10))}
    vm = HadamardVM(fields)
    vm.execute(bytecode)

    print("Hadamard basic test passed.")

def test_compile_pir():
    # Dummy PIR
    fields = [Field(name='theta', type='scalar')]
    operators = [Operator(type='diffusion', trace=[], effect_signature=None)]
    boundary = Boundary(type='none')
    integrator = Integrator(type='Euler')
    step_loop = StepLoop(N=100, dt=0.01)
    pir = PIRProgram(fields=fields, operators=operators, boundary=boundary, integrator=integrator, step_loop=step_loop)

    compiler = HadamardCompiler()
    bytecode = compiler.compile_pir(pir)
    print(f"Compiled bytecode size: {len(bytecode)} bytes")

    print("PIR compilation test passed.")

def benchmark_hadamard():
    # Simple benchmark: compare gradient computation
    field = np.random.rand(50,50,50)  # Smaller for test

    # NumPy
    start = time.time()
    for _ in range(10):
        np_grad = np.gradient(field)
    numpy_time = time.time() - start

    # Hadamard
    assembler = HadamardAssembler()
    assembler.add_instruction('φ', arg1=0, arg2=0)  # Load to reg 0
    assembler.add_instruction('∇', arg1=1, arg2=0)  # Gradient to reg 1
    bytecode = assembler.get_bytecode()
    vm = HadamardVM({'field0': field})

    start = time.time()
    for _ in range(10):
        vm = HadamardVM({'field0': field})  # Reset for each run
        vm.execute(bytecode)
    hadamard_time = time.time() - start

    print(f"NumPy time: {numpy_time:.3f}s, Hadamard time: {hadamard_time:.3f}s")
    speedup = numpy_time / hadamard_time if hadamard_time > 0 else float('inf')
    print(f"Speedup: {speedup:.1f}x")

if __name__ == "__main__":
    test_hadamard_basic()
    test_compile_pir()
    benchmark_hadamard()