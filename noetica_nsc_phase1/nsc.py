from dataclasses import dataclass
import re
import unicodedata
import hashlib
import json
from typing import Union, Tuple, Optional
from . import nsc_diag
from .nsc_parser import parse_program, Program, Span, Phrase, Atom, Group
import os
import zipfile
from zipfile import ZipInfo

def canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(',', ':'))

COMPILER = "nsc 6.1-phase1c"
NSC_VERSION = "6.1-phase1c"

GLYPH_TO_OPCODE = {
    "φ": 1,
    "↻": 2,
    "⊕": 3,
    "⊖": 4,
    "◯": 5,
    "∆": 6,
    "⇒": 7,
    "□": 8,
    "+": 9,
    "-": 10,
    "*": 11,
    "/": 12,
    "∂": 13,
    "∇": 14,
    "∇²": 15,
    "=": 16,
    "(": 17,
    ")": 18,
    # NSC_GR extensions
    "ℋ": 0x21,  # Hamiltonian audit gate
    "𝓜": 0x22,  # Momentum audit gate
    "𝔊": 0x23,  # Gauge enforcement marker
    "𝔇": 0x24,  # Dissipation marker
    "𝔅": 0x25,  # Boundary enforcement marker
    "𝔄": 0x26,  # Accept marker
    "𝔯": 0x27,  # Rollback marker
    "𝕋": 0x28,  # dt arbitration marker
}

@dataclass
class TraceEntry:
    path: str
    span: Span
    file: str
    file_sentence: int
    module_sentence: int

@dataclass
class Bytecode:
    opcodes: list[Union[int, str]]
    trace: list[TraceEntry]

@dataclass
class PDETemplate:
    terms: dict[str, float]  # key: LaTeX term string, value: coefficient
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

@dataclass
class GRPolicy:
    """Hard-typed policy parameters for GR/NR integration."""
    H_max: float
    M_max: float
    R_min: float
    dt_min: float
    dt_max: float
    retry_max: int
    dissip_level: int

    @classmethod
    def default(cls) -> 'GRPolicy':
        """Default GR policy parameters."""
        return cls(
            H_max=1e-8,
            M_max=1e-8,
            R_min=-1e10,
            dt_min=1e-8,
            dt_max=1e-4,
            retry_max=5,
            dissip_level=1
        )

@dataclass
class FlatGlyph:
    glyph: str
    path: str
    span: Span

def compute_source_hash(source: str) -> str:
    return hashlib.sha256(source.encode('utf-8')).hexdigest()

def ast_hash(prog: Program) -> str:
    import dataclasses
    return hashlib.sha256(canonical_json(dataclasses.asdict(prog)).encode('utf-8')).hexdigest()

def bytecode_hash(bytecode: Bytecode) -> str:
    import dataclasses
    return hashlib.sha256(canonical_json(dataclasses.asdict(bytecode)).encode('utf-8')).hexdigest()

def pde_hash(pde: PDETemplate) -> str:
    import dataclasses
    return hashlib.sha256(canonical_json(dataclasses.asdict(pde)).encode('utf-8')).hexdigest()

def policy_hash(policy: GRPolicy) -> str:
    import dataclasses
    return hashlib.sha256(canonical_json(dataclasses.asdict(policy)).encode('utf-8')).hexdigest()

def opcode_table_hash() -> str:
    return hashlib.sha256(canonical_json(GLYPH_TO_OPCODE).encode('utf-8')).hexdigest()

def compute_module_id(source: str, prog: Program, bytecode: Bytecode, pde: PDETemplate, policy: GRPolicy) -> str:
    sh = compute_source_hash(source)
    ah = ast_hash(prog)
    bh = bytecode_hash(bytecode)
    ph = pde_hash(pde)
    pyh = policy_hash(policy)
    oh = opcode_table_hash()
    return f"{sh}-{ah}-{bh}-{ph}-{pyh}-{oh}"

def compute_module_manifest(source: str) -> dict:
    import dataclasses
    prog, flattened, bytecode, pde, policy = nsc_to_pde(source)
    module_id = compute_module_id(source, prog, bytecode, pde, policy)
    return {
        'module_id': module_id,
        'source_hash': compute_source_hash(source),
        'ast_hash': ast_hash(prog),
        'bytecode_hash': bytecode_hash(bytecode),
        'pde_hash': pde_hash(pde),
        'policy_hash': policy_hash(policy),
        'opcode_table_hash': opcode_table_hash(),
        'nsc_version': NSC_VERSION,
        'policy': dataclasses.asdict(policy)  # Include policy parameters
    }

def tokenize(nsc: str) -> list[str]:
    normalized = unicodedata.normalize('NFC', nsc)
    if normalized != nsc:
        raise nsc_diag.NSCError(nsc_diag.E_NONCANONICAL_UNICODE, "Non-canonical Unicode")
    # Pattern to match operators and identifiers/numbers
    pattern = r'(\+|\-|\*|/|∂|∇²|∇|=|\(|\)|\[|\]|φ|⊕|↻|∆|◯|⊖|⇒|□|ℋ|𝓜|𝔊|𝔇|𝔅|𝔄|𝔯|𝕋|\w+|\d+\.?\d*)'
    tokens = re.findall(pattern, normalized)
    tokens = [t for t in tokens if t.strip()]  # remove empty
    if not tokens:
        raise nsc_diag.NSCError(nsc_diag.E_EMPTY_INPUT, "Empty input string")
    # Classify tokens
    for i, t in enumerate(tokens):
        if t == 'import':
            tokens[i] = 'KW_IMPORT'
        elif re.match(r'[A-Za-z_][A-Za-z0-9_]*', t):
            tokens[i] = 'IDENT'
        # else keep as is (operators, numbers, glyphs)
    return tokens

def compile_to_bytecode(flat: list[FlatGlyph], file_path: str = None) -> Bytecode:
    code = []
    trace = []
    for fg in flat:
        glyph = fg.glyph
        if glyph == "⇒":
            opcode = 7
        elif glyph in GLYPH_TO_OPCODE:
            opcode = GLYPH_TO_OPCODE[glyph]
        elif glyph.isalnum() or ('.' in glyph and glyph.replace('.', '').isdigit()):
            opcode = glyph
        else:
            raise nsc_diag.NSCError(nsc_diag.E_UNKNOWN_GLYPH, f"Unknown glyph: {glyph} at {fg.span.start}-{fg.span.end}")
        code.append(opcode)
        # Parse sentence index from path
        parts = fg.path.split('.')
        sentence = int(parts[1]) if parts[0] == 'sentences' and len(parts) > 1 else 0
        file_sentence = sentence
        if file_path is None:
            file = None
            module_sentence = sentence
        else:
            file = file_path
            module_sentence = None  # Will be set later for modules
        trace.append(TraceEntry(path=fg.path, span=fg.span, file=file, file_sentence=file_sentence, module_sentence=module_sentence))
    return Bytecode(opcodes=code, trace=trace)

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

def flatten_phrase_to_list(phrase) -> list[str]:
    result = []
    for item in phrase.items:
        if hasattr(item, 'value'):  # Atom
            result.append(item.value)
        elif hasattr(item, 'delim'):  # Group
            result.extend(flatten_phrase_to_list(item.inner))
    return result

def flatten_program(prog: Program) -> list[str]:
    result = []
    for sentence in prog.sentences:
        result.extend(flatten_phrase_to_list(sentence.lhs))
        if sentence.arrow:
            result.append('⇒')
            if sentence.rhs:
                result.extend(flatten_phrase_to_list(sentence.rhs))
    return result

def flatten_phrase(phrase: Phrase, path_prefix: str) -> list[FlatGlyph]:
    result = []
    for i, item in enumerate(phrase.items):
        if isinstance(item, Atom):
            path = f"{path_prefix}.items.{i}"
            result.append(FlatGlyph(glyph=item.value, path=path, span=Span(start=item.start, end=item.end)))
        elif isinstance(item, Group):
            # recurse without delimiters
            result.extend(flatten_phrase(item.inner, f"{path_prefix}.items.{i}.inner"))
    return result

def flatten_with_trace(prog: Program) -> list[FlatGlyph]:
    result = []
    for i, sentence in enumerate(prog.sentences):
        # flatten lhs
        result.extend(flatten_phrase(sentence.lhs, f"sentences.{i}.lhs"))
        if sentence.arrow:
            # insert FlatGlyph for ⇒ with arrow_span
            path = f"sentences.{i}"
            result.append(FlatGlyph(glyph="⇒", path=path, span=sentence.arrow_span))
            # flatten rhs if present
            if sentence.rhs:
                result.extend(flatten_phrase(sentence.rhs, f"sentences.{i}.rhs"))
    return result

def nsc_to_pde(nsc: str) -> Tuple[Program, list[str], Bytecode, PDETemplate, GRPolicy]:
    tokens = tokenize(nsc)
    prog = parse_program(tokens)
    flat = flatten_with_trace(prog)
    bytecode = compile_to_bytecode(flat)
    pde = assemble_pde(bytecode)
    policy = GRPolicy.default()  # For now, use default; in future, parse from NSC
    return prog, [fg.glyph for fg in flat], bytecode, pde, policy

def create_bundle(src: str, out_path: str) -> None:
    if not os.path.isdir(src):
        raise ValueError(f"Source {src} is not a directory")
    with zipfile.ZipFile(out_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        file_list = []
        nsc_source = None
        for root, dirs, files in os.walk(src):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, src)
                file_list.append((file_path, arcname))
                if arcname.endswith('.nsc') and nsc_source is None:
                    with open(file_path, 'r') as f:
                        nsc_source = f.read()
        file_list.sort(key=lambda x: x[1])  # sort by arcname
        for file_path, arcname in file_list:
            zi = ZipInfo(arcname)
            zi.date_time = (1980, 1, 1, 0, 0, 0)  # deterministic
            zi.external_attr = 0o644 << 16  # unix permissions
            with open(file_path, 'rb') as f:
                zf.writestr(zi, f.read())
        if nsc_source is not None:
            manifest = compute_module_manifest(nsc_source)
            zi = ZipInfo('manifest.json')
            zi.date_time = (1980, 1, 1, 0, 0, 0)
            zi.external_attr = 0o644 << 16
            zf.writestr(zi, canonical_json(manifest))

def verify_bundle(bundle_path: str) -> Tuple[dict, bool]:
    verify_dict = {}
    verified = True
    try:
        with zipfile.ZipFile(bundle_path, 'r') as zf:
            nsc_source = None
            manifest_data = None
            for zi in zf.infolist():
                if not zi.filename.endswith('/'):  # not a directory
                    with zf.open(zi) as f:
                        data = f.read()
                        h = hashlib.sha256(data).hexdigest()
                        verify_dict[zi.filename] = h
                        if zi.filename.endswith('.nsc') and nsc_source is None:
                            nsc_source = data.decode('utf-8')
                        elif zi.filename == 'manifest.json':
                            manifest_data = data.decode('utf-8')
            if manifest_data and nsc_source:
                stored_manifest = json.loads(manifest_data)
                current_manifest = compute_module_manifest(nsc_source)
                if stored_manifest['module_id'] != current_manifest['module_id']:
                    verified = False
                    verify_dict['verification_error'] = 'module_id mismatch'
            elif not manifest_data:
                verified = False
                verify_dict['verification_error'] = 'no manifest.json'
    except Exception as e:
        verified = False
        verify_dict['error'] = str(e)
    return verify_dict, verified