from .ast import Program
from .flatten import flatten_with_trace
from ..solver.pir import PIRProgram, Operator, TraceEntry, Field, Boundary, Integrator, StepLoop

def lower_to_pir(prog: Program) -> PIRProgram:
    flat = flatten_with_trace(prog)
    operators = []
    boundary_type = 'none'
    integrator_type = 'Euler'  # default
    for fg in flat:
        glyph = fg.glyph
        origin = f"{fg.path}:{fg.span.start}-{fg.span.end}"
        if glyph == '◯':
            operators.append(Operator(type='diffusion', trace=[TraceEntry(glyph=glyph, origin=origin)]))
        elif glyph == '⊕':
            operators.append(Operator(type='source', trace=[TraceEntry(glyph=glyph, origin=origin)]))
        elif glyph == '⊖':
            operators.append(Operator(type='sink', trace=[TraceEntry(glyph=glyph, origin=origin)]))
        elif glyph == '↻':
            operators.append(Operator(type='curvature_coupling', trace=[TraceEntry(glyph=glyph, origin=origin)]))
        elif glyph == '∆':
            operators.append(Operator(type='damping', trace=[TraceEntry(glyph=glyph, origin=origin)]))
        elif glyph == '□':
            boundary_type = 'dirichlet'
        elif glyph == '⇒':
            integrator_type = 'RK2'  # step marker, perhaps better integrator
    fields = [Field(name='theta', type='scalar')]
    boundary = Boundary(type=boundary_type)
    integrator = Integrator(type=integrator_type)
    step_loop = StepLoop(N=1000, dt=0.01)  # defaults, perhaps adjustable
    return PIRProgram(fields=fields, operators=operators, boundary=boundary, integrator=integrator, step_loop=step_loop)