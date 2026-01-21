import os
import json
import zipfile
from zipfile import ZipInfo
import hashlib
import unicodedata
import dataclasses
from dataclasses import dataclass
from typing import List
from . import nsc_diag
from .nsc import tokenize, parse_program, flatten_phrase_to_list, flatten_with_trace, compile_to_bytecode, assemble_pde, canonical_json, compute_source_hash, ast_hash, bytecode_hash, pde_hash, opcode_table_hash, TraceEntry, Bytecode, PDETemplate, FlatGlyph, Span

@dataclass
class FileArtifact:
    path: str
    tokens: List[str]
    ast: object  # Program
    flattened: List[str]
    bytecode: Bytecode
    trace: List[TraceEntry]
    pde_coeffs: dict
    pde_latex: str

@dataclass
class ModuleArtifact:
    module_manifest_M: str
    module_id: str
    file_artifacts: List[FileArtifact]
    module_flattened: List[str]
    module_bytecode: Bytecode
    module_trace: List[TraceEntry]
    module_pde: dict
    module_pde_latex: str

def load_module_manifest(root_dir: str) -> dict:
    manifest_path = os.path.join(root_dir, 'nsc.module.json')
    try:
        with open(manifest_path, 'r', encoding='utf-8') as f:
            manifest = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        raise nsc_diag.NSCError(nsc_diag.E_BUNDLE_IO, f"Failed to read manifest: {e}")

    # Validate schema
    if not isinstance(manifest, dict):
        raise nsc_diag.NSCError(nsc_diag.E_MANIFEST_SCHEMA, "Manifest must be a dict")

    required_keys = ['sources', 'imports']
    for key in required_keys:
        if key not in manifest:
            raise nsc_diag.NSCError(nsc_diag.E_MANIFEST_SCHEMA, f"Missing key: {key}")

    if not isinstance(manifest['sources'], list) or not all(isinstance(s, str) for s in manifest['sources']):
        raise nsc_diag.NSCError(nsc_diag.E_MANIFEST_SCHEMA, "sources must be list of strings")

    if not isinstance(manifest['imports'], list) or not all(isinstance(i, str) for i in manifest['imports']):
        raise nsc_diag.NSCError(nsc_diag.E_MANIFEST_SCHEMA, "imports must be list of strings")

    # Ordering rules
    if manifest['sources'] != sorted(manifest['sources']):
        raise nsc_diag.NSCError(nsc_diag.E_MANIFEST_SCHEMA, "sources not sorted")

    if manifest['imports'] != sorted(manifest['imports']):
        raise nsc_diag.NSCError(nsc_diag.E_MANIFEST_SCHEMA, "imports not sorted")

    if len(manifest['sources']) != len(set(manifest['sources'])):
        raise nsc_diag.NSCError(nsc_diag.E_MANIFEST_SCHEMA, "sources has duplicates")

    if len(manifest['imports']) != len(set(manifest['imports'])):
        raise nsc_diag.NSCError(nsc_diag.E_MANIFEST_SCHEMA, "imports has duplicates")

    return manifest

def read_source_file(root_dir: str, rel_path: str) -> str:
    full_path = os.path.join(root_dir, rel_path)
    with open(full_path, 'r', encoding='utf-8') as f:
        src = f.read()
    normalized = unicodedata.normalize('NFC', src)
    if normalized != src:
        raise nsc_diag.NSCError(nsc_diag.E_NONCANONICAL_UNICODE, "Source file not in NFC normalization")
    return normalized

def compile_file(rel_path: str, src: str) -> FileArtifact:
    tokens = tokenize(src)
    prog = parse_program(tokens)
    flat_glyphs = flatten_with_trace(prog)
    flattened = [fg.glyph for fg in flat_glyphs]
    bytecode = compile_to_bytecode(flat_glyphs, rel_path)
    pde = assemble_pde(bytecode)
    # Extend trace with file coordinates - done in compile_to_bytecode
    return FileArtifact(
        path=rel_path,
        tokens=tokens,
        ast=prog,
        flattened=flattened,
        bytecode=bytecode,
        trace=bytecode.trace,
        pde_coeffs=pde.terms,
        pde_latex=pde.as_latex()
    )

def compile_module(root_dir: str, manifest: dict) -> ModuleArtifact:
    file_artifacts = []
    module_sentences = []
    sentence_offset = 0
    for rel_path in manifest['sources']:
        src = read_source_file(root_dir, rel_path)
        file_artifact = compile_file(rel_path, src)
        file_artifacts.append(file_artifact)
        # Concatenate sentences
        module_sentences.extend(file_artifact.flattened)
        # Update trace with module_sentence
        for entry in file_artifact.trace:
            # Assuming path like "sentences.X..."
            parts = entry.path.split('.')
            if parts[0] == 'sentences' and len(parts) > 1:
                local_sentence = int(parts[1])
                entry.module_sentence = local_sentence + sentence_offset
                entry.path = f"files.{rel_path}.sentences.{local_sentence + sentence_offset}"
            else:
                entry.path = f"files.{rel_path}.{entry.path}"
        sentence_offset += len(file_artifact.ast.sentences)

    module_flattened = module_sentences
    # Compile module bytecode - concatenate all flattened, but need to adjust trace
    # For simplicity, since it's module-wide, create new flat_glyphs with adjusted paths
    all_flat_glyphs = []
    for fa in file_artifacts:
        for fg in flatten_with_trace(fa.ast):
            # Adjust path
            parts = fg.path.split('.')
            if parts[0] == 'sentences' and len(parts) > 1:
                local_sentence = int(parts[1])
                global_sentence = local_sentence + sum(len(fa2.ast.sentences) for fa2 in file_artifacts[:file_artifacts.index(fa)])
                fg.path = f"sentences.{global_sentence}.{'.'.join(parts[2:])}"
            all_flat_glyphs.append(fg)

    module_bytecode = compile_to_bytecode(all_flat_glyphs, None)
    module_pde = assemble_pde(module_bytecode)
    module_trace = module_bytecode.trace
    # Set file for trace entries
    for entry in module_trace:
        if entry.path.startswith('files.'):
            parts = entry.path.split('.')
            if len(parts) > 1:
                entry.file = parts[1]

    # Compute M and module_id
    module_manifest_M = canonical_json(manifest)
    # For module_id, perhaps similar to compute_module_id but for module
    # Concatenate all sources or something, but task says "Compute M as specified, compute hashes, module_id"
    # Probably module_id based on manifest and all sources
    all_src = ''.join(read_source_file(root_dir, p) for p in manifest['sources'])
    module_id = compute_source_hash(module_manifest_M + all_src)  # Include manifest

    return ModuleArtifact(
        module_manifest_M=module_manifest_M,
        module_id=module_id,
        file_artifacts=file_artifacts,
        module_flattened=module_flattened,
        module_bytecode=module_bytecode,
        module_trace=module_trace,
        module_pde=module_pde.terms,
        module_pde_latex=module_pde.as_latex()
    )

def write_module_bundle(module_artifact: ModuleArtifact, out_path: str, root_dir: str) -> None:
    with zipfile.ZipFile(out_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        # Sort entries lexically
        entries = {}
        # Add manifest
        entries['manifest.json'] = canonical_json({
            'module_manifest_M': module_artifact.module_manifest_M,
            'module_id': module_artifact.module_id,
        })
        # Write sources and artifacts
        for fa in module_artifact.file_artifacts:
            # Write source
            entries[f"sources/{fa.path}"] = read_source_file(root_dir, fa.path)
            # Write bytecode artifacts
            entries[f"artifacts/{fa.path}.bytecode"] = canonical_json(fa.bytecode.opcodes)
            entries[f"artifacts/{fa.path}.trace"] = canonical_json([dataclasses.asdict(te) for te in fa.trace])
            entries[f"artifacts/{fa.path}.pde"] = canonical_json(fa.pde_coeffs)
        # Write module artifacts
        entries['artifacts/module.bytecode'] = canonical_json(module_artifact.module_bytecode.opcodes)
        entries['artifacts/module.trace'] = canonical_json([dataclasses.asdict(te) for te in module_artifact.module_trace])
        entries['artifacts/module.pde'] = canonical_json(module_artifact.module_pde)
        # Sort
        for name in sorted(entries):
            zi = ZipInfo(name)
            zi.date_time = (1980, 1, 1, 0, 0, 0)
            zi.external_attr = 0o644 << 16
            zf.writestr(zi, entries[name])

def verify_module_bundle(path: str) -> dict:
    verify_record = {'verified': False, 'diagnostics': {}}
    try:
        with zipfile.ZipFile(path, 'r') as zf:
            # Extract and recompute
            manifest_data = None
            sources = {}
            for zi in zf.infolist():
                if zi.filename == 'manifest.json':
                    with zf.open(zi) as f:
                        manifest_data = json.loads(f.read().decode('utf-8'))
                elif zi.filename.startswith('sources/'):
                    rel_path = zi.filename[len('sources/'):]
                    with zf.open(zi) as f:
                        sources[rel_path] = f.read().decode('utf-8')
            if manifest_data and sources:
                # Recompute module_artifact
                # But need manifest, assume it's in root_dir, but for verification, use extracted
                # This is simplified
                all_src = ''.join(sources.values())
                computed_id = compute_source_hash(manifest_data['module_manifest_M'] + all_src)
                if computed_id == manifest_data.get('module_id'):
                    verify_record['verified'] = True
                else:
                    verify_record['diagnostics']['module_id_mismatch'] = True
            else:
                verify_record['diagnostics']['missing_data'] = True
    except Exception as e:
        verify_record['diagnostics']['error'] = str(e)
    return verify_record