#!/usr/bin/env python3
import json, re, sys, hashlib

def canon(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

def sha256_hex(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def parse_struct(text):
    # Simple parser for struct { field: type, ... }
    text = text.strip()
    if not text.startswith('struct {') or not text.endswith('}'):
        raise ValueError("Bad struct syntax")
    inner = text[8:-1].strip()
    fields = {}
    for field in inner.split(','):
        field = field.strip()
        if ':' not in field:
            continue
        name, typ = field.split(':', 1)
        name = name.strip()
        typ = typ.strip()
        fields[name] = typ
    return fields

def compile_nsc_text(src: str) -> dict:
    # strip comments
    src = re.sub(r"//.*?$", "", src, flags=re.MULTILINE)
    src = src.strip()

    m_mod = re.search(r"nsc\.module\s+([A-Za-z0-9_.-]+)\s*;", src)
    if not m_mod:
        raise ValueError("Missing `nsc.module <name>;`")
    module = m_mod.group(1)

    # Find op block
    m_op = re.search(r"op\s+([A-Za-z_]\w*)\s*\{(.*)\}\s*\Z", src, flags=re.DOTALL)
    if not m_op:
        raise ValueError("Missing `op <name> { ... }` block")
    op_name = m_op.group(1)
    body = m_op.group(2)

    def grab(pattern, required=True):
        mm = re.search(pattern, body)
        if not mm and required:
            raise ValueError(f"Missing required field matching: {pattern}")
        return mm.group(1).strip() if mm else None

    lane = grab(r"lane\s*:\s*([A-Za-z]+\.[A-Za-z]+\.[A-Za-z]+)\s*;")
    effects = grab(r"effects\s*:\s*read\[(.*?)\]\s*,\s*write\[(.*?)\]\s*;", required=True)
    mm_eff = re.search(r"effects\s*:\s*read\[(.*?)\]\s*,\s*write\[(.*?)\]\s*;", body)
    read_eff = [x.strip() for x in mm_eff.group(1).split(",") if x.strip()]
    write_eff = [x.strip() for x in mm_eff.group(2).split(",") if x.strip()]

    bc = grab(r"bc\s*:\s*([A-Za-z_]\w*)\s*;")
    dim = int(grab(r"dim\s*:\s*(\d+)\s*;"))

    # Inputs / outputs - now structs
    m_in = re.search(r"in\s+([A-Za-z_]\w*)\s*:\s*struct\s*\{(.*?)\}\s*@([A-Za-z_]\w*)\s*;", body, re.DOTALL)
    m_out = re.search(r"out\s+([A-Za-z_]\w*)\s*:\s*struct\s*\{(.*?)\}\s*@([A-Za-z_]\w*)\s*;", body, re.DOTALL)
    if not (m_in and m_out):
        raise ValueError("Missing `in fields: struct { ... } @S_PHY;` or `out rhs_bundle: ...`")

    in_name, in_struct_text, in_comp = m_in.group(1), m_in.group(2), m_in.group(3)
    out_name, out_struct_text, out_comp = m_out.group(1), m_out.group(2), m_out.group(3)

    in_struct = parse_struct('struct {' + in_struct_text + '}')
    out_struct = parse_struct('struct {' + out_struct_text + '}')

    # Params
    params = {}
    m_lambda = re.search(r"param\s+lambda_val\s*:\s*f64\s*=\s*([0-9.eE+-]+)\s*;", body)
    params['lambda_val'] = float(m_lambda.group(1)) if m_lambda else 0.0
    m_sources = re.search(r"param\s+sources_enabled\s*:\s*bool\s*=\s*(true|false)\s*;", body)
    params['sources_enabled'] = m_sources.group(1) == 'true' if m_sources else False

    # Expression line
    expr_pattern = r"([A-Za-z_]\w*)\s*=\s*compute_gr_rhs\(\s*([A-Za-z_]\w*)\s*,\s*lambda_val\s*,\s*sources_enabled\s*\)\s*;"
    mm_expr = re.search(expr_pattern, body)
    if not mm_expr:
        raise ValueError("Expression must be: rhs_bundle = compute_gr_rhs(fields, lambda_val, sources_enabled);")
    lhs, rhs_arg = mm_expr.group(1), mm_expr.group(2)
    if lhs != out_name or rhs_arg != in_name:
        raise ValueError("Expression mismatch")

    ir = {
        "schema": "nsc_ir_v0.1",
        "module": module,
        "op": {
            "name": op_name,
            "lane": lane,
            "effects": {"read": read_eff, "write": write_eff},
            "bc": bc,
            "dim": dim,
            "in": {"name": in_name, "struct": in_struct, "compartment": in_comp},
            "out": {"name": out_name, "struct": out_struct, "compartment": out_comp},
            "params": params,
            "expr": {"kind": "gr_rhs_bundle", "fields": in_name, "lambda_val": "lambda_val", "sources_enabled": "sources_enabled", "out": out_name},
        },
    }
    ir_bytes = canon(ir).encode("utf-8")
    ir["ir_hash"] = sha256_hex(b"NSCIR\0" + ir_bytes)
    return ir

def main():
    if len(sys.argv) != 3:
        print("Usage: python3 nsc_compile_min.py <in.nsc> <out.nscir.json>")
        raise SystemExit(2)
    in_path, out_path = sys.argv[1], sys.argv[2]
    with open(in_path, "r", encoding="utf-8") as f:
        src = f.read()
    ir = compile_nsc_text(src)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(ir, f, indent=2, sort_keys=True)
    print(f"Wrote {out_path}")
    print(f"ir_hash={ir['ir_hash']}")

if __name__ == "__main__":
    main()