from dataclasses import dataclass
from typing import Dict
from .compile_bc import Bytecode

@dataclass
class PDETemplate:
    terms: Dict[str, float]  # key: LaTeX term string, value: coefficient
    boundary: str = "none"

    def as_latex(self) -> str:
        left = "\\partial_t \\theta"
        if not self.terms:
            right = "0"
        else:
            right_parts = []
            for term, coeff in self.terms.items():
                if coeff == 0:
                    continue
                if coeff > 0:
                    right_parts.append(f" + {coeff}\\\,{term}")
                else:
                    right_parts.append(f" - {-coeff}\\\,{term}")
            right = "".join(right_parts)
        return f"{left}{right} = 0"

def assemble_pde(bytecode: Bytecode) -> PDETemplate:
    opcodes = bytecode.opcodes
    if 16 in opcodes:  # EQ present, old logic
        terms = {}
        i = 0
        while i < len(opcodes):
            if opcodes[i] == 16:  # EQ
                i += 1
                break
            i += 1
        coeff = 1.0
        while i < len(opcodes):
            opc = opcodes[i]
            if opc == 9:  # ADD
                pass
            elif opc == 10:  # SUB
                coeff = -coeff
            elif isinstance(opc, str) and (opc.isdigit() or '.' in opc):
                coeff = float(opc)
            elif opc == 11:  # MUL
                pass
            elif opc == 15:  # LAP
                i += 1
                if i < len(opcodes):
                    term = f"\\nabla^2 {opcodes[i]}"
                    terms[term] = terms.get(term, 0) + coeff
                    coeff = 1.0
            elif opc == 14:  # GRAD
                i += 1
                if i < len(opcodes):
                    term = f"\\nabla {opcodes[i]}"
                    terms[term] = terms.get(term, 0) + coeff
                    coeff = 1.0
            elif opc == 13:  # ∂
                i += 1
                if i < len(opcodes):
                    term = f"\\partial_t {opcodes[i]}"
                    terms[term] = terms.get(term, 0) + coeff
                    coeff = 1.0
            else:
                if isinstance(opc, str):
                    term = opc
                    terms[term] = terms.get(term, 0) + coeff
                    coeff = 1.0
            i += 1
        boundary = 'none'
    else:  # new logic
        terms = {}
        if 2 in opcodes:  # ↻
            terms['\\nabla^2\\theta'] = -2.0
        if 6 in opcodes:  # ∆
            terms['\\partial_t\\theta'] = 0.1
        if 5 in opcodes:  # ◯
            terms['R\\theta'] = 1.0
        boundary = '□' if 8 in opcodes else 'none'
    return PDETemplate(terms, boundary)