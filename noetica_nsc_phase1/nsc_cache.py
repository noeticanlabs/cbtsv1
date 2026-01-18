import os
import json
import zipfile
import shutil
import io

from .nsc_diag import NSCError
from .nsc import verify_bundle, compute_module_id

def cache_root():
    return os.path.expanduser("~/.cache/nsc")

def cache_dir(module_id):
    return os.path.join(cache_root(), module_id)

def cache_has_verified(module_id):
    d = cache_dir(module_id)
    manifest_path = os.path.join(d, "manifest.json")
    if not os.path.exists(manifest_path):
        return False
    try:
        with open(manifest_path, 'r') as f:
            stored_manifest = json.load(f)
        return True
    except:
        return False

def cache_load(module_id):
    """Load the manifest from cache."""
    d = cache_dir(module_id)
    manifest_path = os.path.join(d, "manifest.json")
    with open(manifest_path, 'r') as f:
        return json.load(f)

def cache_validate_loaded(module_id, loaded):
    """Validate the loaded manifest."""
    return loaded.get('module_id') == module_id

def cache_store_from_bundle(bundle_path, module_id):
    verify_dict, verified = verify_bundle(bundle_path)
    if not verified:
        raise NSCError("Bundle not verified", "cache_store_from_bundle")
    # Get module_id from bundle
    with zipfile.ZipFile(bundle_path, 'r') as zf:
        with zf.open('manifest.json') as f:
            manifest = json.loads(f.read().decode('utf-8'))
    if manifest['module_id'] != module_id:
        raise NSCError("Module ID mismatch", "cache_store_from_bundle")
    # Extract to cache_dir
    d = cache_dir(module_id)
    os.makedirs(d, exist_ok=True)
    with zipfile.ZipFile(bundle_path, 'r') as zf:
        zf.extractall(d)

def get_cached_module_bundle(module_id: str) -> bytes | None:
    path = os.path.join(cache_dir(module_id), "bundle.nscb")
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return f.read()
    return None

def set_cached_module_bundle(module_id: str, bundle_bytes: bytes) -> None:
    d = cache_dir(module_id)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, "bundle.nscb")
    with open(path, 'wb') as f:
        f.write(bundle_bytes)
    # Optionally extract individual files for faster access
    with zipfile.ZipFile(io.BytesIO(bundle_bytes), 'r') as zf:
        zf.extractall(d)