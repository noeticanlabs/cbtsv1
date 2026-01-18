import json
import hashlib
import os

def load_manifest(path):
    """Load manifest from JSON file."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Manifest file not found: {path}")
    with open(path, 'r') as f:
        return json.load(f)

def validate_manifest(manifest):
    """Validate the manifest against the schema."""
    required_keys = ['name', 'version', 'compiler', 'nsc_version', 'nllc_version', 'sources', 'imports']
    for key in required_keys:
        if key not in manifest:
            raise ValueError(f"Missing required key: {key}")

    # Validate types
    if not isinstance(manifest['name'], str):
        raise ValueError("name must be a string")
    if not isinstance(manifest['version'], str):
        raise ValueError("version must be a string")
    if not isinstance(manifest['compiler'], str):
        raise ValueError("compiler must be a string")
    if not isinstance(manifest['nsc_version'], str):
        raise ValueError("nsc_version must be a string")
    if not isinstance(manifest['nllc_version'], str):
        raise ValueError("nllc_version must be a string")
    if not isinstance(manifest['sources'], list):
        raise ValueError("sources must be a list")
    if not isinstance(manifest['imports'], list):
        raise ValueError("imports must be a list")

    # Validate sources
    for src in manifest['sources']:
        if not isinstance(src, dict) or 'path' not in src or 'lang' not in src:
            raise ValueError("Each source must have 'path' and 'lang'")
        if not isinstance(src['path'], str):
            raise ValueError("source path must be a string")
        if src['lang'] not in ['nllc', 'nsc']:
            raise ValueError("source lang must be 'nllc' or 'nsc'")

    # Sources must be sorted by path
    sorted_sources = sorted(manifest['sources'], key=lambda x: x['path'])
    if manifest['sources'] != sorted_sources:
        raise ValueError("sources must be sorted by path")

    # Validate imports
    for imp in manifest['imports']:
        if not isinstance(imp, dict) or 'name' not in imp or 'module_id' not in imp or 'bundle_path' not in imp:
            raise ValueError("Each import must have 'name', 'module_id', 'bundle_path'")
        if not isinstance(imp['name'], str):
            raise ValueError("import name must be a string")
        if not isinstance(imp['module_id'], str):
            raise ValueError("import module_id must be a string")
        if not isinstance(imp['bundle_path'], str):
            raise ValueError("import bundle_path must be a string")

    # Imports must be sorted by name
    sorted_imports = sorted(manifest['imports'], key=lambda x: x['name'])
    if manifest['imports'] != sorted_imports:
        raise ValueError("imports must be sorted by name")

    # Optional entry
    if 'entry' in manifest:
        entry = manifest['entry']
        if not isinstance(entry, dict) or 'path' not in entry or 'lang' not in entry:
            raise ValueError("entry must have 'path' and 'lang'")
        if not isinstance(entry['path'], str):
            raise ValueError("entry path must be a string")
        if entry['lang'] not in ['nllc', 'nsc']:
            raise ValueError("entry lang must be 'nllc' or 'nsc'")

    return True

def compute_module_id(manifest, dep_closure_hash, policy_hash):
    """Compute module ID based on manifest, dependency closure hash, and policy hash."""
    # Create a string representation for hashing
    parts = [
        manifest['name'],
        manifest['version'],
        json.dumps(manifest['sources'], sort_keys=True),
        json.dumps(manifest['imports'], sort_keys=True),
        dep_closure_hash,
        policy_hash
    ]
    combined = '|'.join(parts)
    return hashlib.sha256(combined.encode('utf-8')).hexdigest()