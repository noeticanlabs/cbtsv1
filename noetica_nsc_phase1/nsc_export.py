from noetica_nsc_phase1 import nsc
from noetica_nsc_phase1 import nsc_diag
from .nsc_parser import Program

def export_symbolic(src, prog: Program, flattened: list[str], bc, tpl) -> dict:
    """
    Export symbolic representation to canonical JSON schema dict.
    """
    try:
        return {
            "nsc_version": "6.1-phase1c",
            "ast": prog.to_dict(),
            "flattened": flattened,
            "determinism": {
                "opcode_table": bc.opcodes,
                "compose_rule": "sequential",
                "whitespace_policy": "ignore"
            }
        }
    except Exception as e:
        raise nsc_diag.NSCError(nsc_diag.E_EXPORT_IO, f"Error in symbolic export: {str(e)}")