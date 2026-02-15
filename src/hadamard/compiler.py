import json
from solver.pir import PIRProgram, Operator
from .assembler import HadamardAssembler

class HadamardCompiler:
    def __init__(self):
        self.field_map = {}  # Map field names to indices

    def compile_from_ir(self, ir_path: str) -> bytes:
        """Compile .nscir.json to Hadamard bytecode."""
        with open(ir_path, 'r') as f:
            ir = json.load(f)
        if ir.get('schema') != 'nsc_ir_v0.1':
            raise ValueError("Unsupported IR schema")
        op = ir['op']
        if op['expr']['kind'] != 'gr_rhs_bundle':
            raise ValueError("Unsupported op kind")
        # For now, generate placeholder bytecode that calls the RHS computation
        # In full implementation, parse the expr and fields to generate bytecode
        assembler = HadamardAssembler()
        # Placeholder: add some instructions
        assembler.add_instruction('ricci', arg1=0)  # Compute Ricci
        assembler.add_instruction('lie', arg1=1, arg2=2)  # Lie derivative
        # etc.
        return assembler.get_bytecode()

    def compile_pir(self, pir: PIRProgram) -> bytes:
        assembler = HadamardAssembler()
        # Assign indices to fields
        for i, field in enumerate(pir.fields):
            self.field_map[field.name] = i

        # Dead code elimination: find used fields
        used_fields = set()
        for op in pir.operators:
            self.analyze_operator(op, used_fields)

        # Compile only ops that affect used fields
        optimized_ops = [op for op in pir.operators if self.affects_used(op, used_fields)]

        for op in optimized_ops:
            self.compile_operator(op, assembler)

        return assembler.get_bytecode()

    def analyze_operator(self, op: Operator, used_fields: set):
        # Mark fields used in this op
        if hasattr(op, 'field'):
            used_fields.add(op.field)
        # Add more based on op type

    def affects_used(self, op: Operator, used_fields: set) -> bool:
        # Check if op affects a used field
        if hasattr(op, 'target_field'):
            return op.target_field in used_fields
        return True  # Default include

    def compile_operator(self, op: Operator, assembler: HadamardAssembler):
        if op.type == 'diffusion':
            # ∇² on field
            field_idx = self.field_map.get('theta', 0)  # Placeholder
            assembler.add_instruction('∇²', arg1=field_idx)
        elif op.type == 'source':
            # ⊕
            assembler.add_instruction('⊕', arg1=0, arg2=1)  # Placeholder indices
        elif op.type == 'sink':
            assembler.add_instruction('⊖', arg1=0, arg2=1)
        elif op.type == 'curvature_coupling':
            assembler.add_instruction('↻', arg1=0, arg2=1)
        elif op.type == 'damping':
            assembler.add_instruction('∆', arg1=0)
        # Add more mappings as per spec