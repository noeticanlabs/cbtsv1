from dataclasses import dataclass, asdict
import hashlib
import json
from typing import Dict, List, Union, Any


@dataclass
class EffectSignature:
    read_compartments: List[str]  # List of compartments read from, e.g., ['fields', 'geometry']
    write_compartments: List[str]  # List of compartments written to, e.g., ['fields']
    determinism: str  # 'pure', 'reads_only', 'writes'
    requires_audit_before_commit: bool  # True if audit needed before committing changes


# Predefined effect signatures for GR operations
GR_OP_SIGNATURES = {
    'evolve': EffectSignature(
        read_compartments=['fields', 'geometry', 'constraints', 'gauge'],
        write_compartments=['fields'],
        determinism='writes',
        requires_audit_before_commit=True  # Evolve modifies state significantly, audit needed
    ),
    'solve_constraints': EffectSignature(
        read_compartments=['fields', 'geometry'],
        write_compartments=['constraints'],
        determinism='writes',
        requires_audit_before_commit=False  # Solving constraints is deterministic, no audit needed
    ),
    'update_geometry': EffectSignature(
        read_compartments=['fields'],
        write_compartments=['geometry'],
        determinism='writes',
        requires_audit_before_commit=False  # Geometry update is deterministic from fields
    ),
    'diffusion': EffectSignature(
        read_compartments=['fields'],
        write_compartments=['fields'],
        determinism='writes',
        requires_audit_before_commit=False
    ),
    'source': EffectSignature(
        read_compartments=['fields'],
        write_compartments=['fields'],
        determinism='writes',
        requires_audit_before_commit=False
    ),
    'sink': EffectSignature(
        read_compartments=['fields'],
        write_compartments=['fields'],
        determinism='writes',
        requires_audit_before_commit=False
    ),
    'curvature_coupling': EffectSignature(
        read_compartments=['fields', 'geometry'],
        write_compartments=['fields'],
        determinism='writes',
        requires_audit_before_commit=False
    ),
}


@dataclass
class TraceEntry:
    glyph: str
    origin: str
    # Add other fields as needed for NSC glyph origins


@dataclass
class Operator:
    type: str  # 'diffusion', 'source', 'sink', 'curvature_coupling', 'evolve', 'solve_constraints', 'update_geometry'
    # Additional fields specific to operator, e.g., coefficients, etc.
    trace: List[TraceEntry]
    effect_signature: EffectSignature  # New field for twin-track execution safety


@dataclass
class Field:
    name: str
    type: str  # 'scalar' or 'array'
    # Additional metadata if needed


@dataclass
class Boundary:
    type: str  # 'none' or 'dirichlet'
    # Dirichlet conditions if applicable


@dataclass
class Integrator:
    type: str  # 'Euler' or 'RK2'


@dataclass
class StepLoop:
    N: int
    dt: float


@dataclass
class PIRProgram:
    fields: List[Field]
    operators: List[Operator]
    boundary: Boundary
    integrator: Integrator
    step_loop: StepLoop

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PIRProgram':
        # Assuming nested dataclasses can be reconstructed, might need custom logic
        fields = [Field(**f) for f in data['fields']]
        operators = []
        for op_data in data['operators']:
            effect_sig = EffectSignature(**op_data['effect_signature']) if 'effect_signature' in op_data else EffectSignature([], [], 'pure', False)
            op = Operator(type=op_data['type'], trace=[TraceEntry(**t) for t in op_data['trace']], effect_signature=effect_sig)
            operators.append(op)
        boundary = Boundary(**data['boundary'])
        integrator = Integrator(**data['integrator'])
        step_loop = StepLoop(**data['step_loop'])
        return cls(fields=fields, operators=operators, boundary=boundary, integrator=integrator, step_loop=step_loop)

    def canonical_to_dict(self) -> Dict[str, Any]:
        # Convert to dict and sort keys recursively
        def sort_dict(d):
            if isinstance(d, dict):
                return {k: sort_dict(v) for k, v in sorted(d.items())}
            elif isinstance(d, list):
                return [sort_dict(item) for item in d]
            else:
                return d
        return sort_dict(self.to_dict())

    def hash_pir(self) -> str:
        canonical = self.canonical_to_dict()
        json_str = json.dumps(canonical, sort_keys=True)
        return hashlib.sha256(json_str.encode()).hexdigest()