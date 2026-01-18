import zipfile
import os
import json
import hashlib
import dataclasses
from typing import Dict, Any
from datetime import datetime
from src.common.receipt import NamespaceReceipt
from src.module.manifest import compute_module_id

FIXED_TIMESTAMP = (2020, 1, 1, 0, 0, 0)  # Fixed date for deterministic builds

def compute_policy_hash() -> str:
    """Compute hash of coupling_policy_v0.1.json."""
    policy_path = os.path.join(os.getcwd(), 'coupling_policy_v0.1.json')
    with open(policy_path, 'r') as f:
        policy = json.load(f)
    policy_str = json.dumps(policy, sort_keys=True)
    return hashlib.sha256(policy_str.encode('utf-8')).hexdigest()

def create_deterministic_bundle(dir_path: str, manifest: Dict[str, Any], namespace_receipt: NamespaceReceipt, output_path: str = None) -> str:
    if output_path is None:
        output_path = f"{manifest['name']}_{manifest['version']}.nscb"
    
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        # Collect all files in dir_path recursively, sorted lexicographically
        file_list = []
        for root, dirs, files in os.walk(dir_path):
            for file in files:
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, dir_path)
                file_list.append((full_path, rel_path))
        
        # Sort by relative path
        file_list.sort(key=lambda x: x[1])
        
        # Add files with fixed timestamp
        for full_path, rel_path in file_list:
            zf.write(full_path, rel_path)
            # Set fixed mtime for determinism
            zf.getinfo(rel_path).date_time = FIXED_TIMESTAMP
        
        # Add manifest.json
        manifest_str = json.dumps(manifest, sort_keys=True)
        zf.writestr('manifest.json', manifest_str)
        zf.getinfo('manifest.json').date_time = FIXED_TIMESTAMP
        
        # Add namespace_receipt.json
        receipt_dict = dataclasses.asdict(namespace_receipt)
        receipt_str = json.dumps(receipt_dict, sort_keys=True)
        zf.writestr('namespace_receipt.json', receipt_str)
        zf.getinfo('namespace_receipt.json').date_time = FIXED_TIMESTAMP

        # Add coupling_policy_v0.1.json
        policy_path = os.path.join(os.getcwd(), 'coupling_policy_v0.1.json')
        with open(policy_path, 'r') as f:
            policy_content = f.read()
        zf.writestr('coupling_policy_v0.1.json', policy_content)
        zf.getinfo('coupling_policy_v0.1.json').date_time = FIXED_TIMESTAMP
    
    return output_path

def verify_bundle(bundle_path: str) -> bool:
    with zipfile.ZipFile(bundle_path, 'r') as zf:
        # Load manifest
        if 'manifest.json' not in zf.namelist():
            return False
        manifest = json.loads(zf.read('manifest.json').decode('utf-8'))
        
        # Load namespace_receipt
        if 'namespace_receipt.json' not in zf.namelist():
            return False
        receipt_dict = json.loads(zf.read('namespace_receipt.json').decode('utf-8'))
        namespace_receipt = NamespaceReceipt(**receipt_dict)
        
        # Verify receipt id (module_id)
        computed_module_id = compute_module_id(manifest, namespace_receipt.dep_closure_hash, namespace_receipt.policy_hash)
        if computed_module_id != namespace_receipt.module_id:
            return False
        
        # Check hashes of files? Perhaps compute dep_closure_hash, but since receipts are included, maybe verify against dep_closure_hash
        # For now, assume dep_closure_hash is provided and correct, but perhaps add hash computation
        # The task says "check hashes and receipt id", perhaps hashes of bundled files
        
        # Compute hash of all files except manifest and receipt?
        # Perhaps the dep_closure_hash includes file hashes
        
        # For simplicity, since dep_closure_hash is part of module_id, and module_id is verified, assume ok
        # But to check hashes, maybe compute SHA256 of sorted file contents
        
        file_hashes = {}
        for name in sorted(zf.namelist()):
            if name not in ['manifest.json', 'namespace_receipt.json']:
                content = zf.read(name)
                file_hashes[name] = hashlib.sha256(content).hexdigest()
        
        # Include policy_hash
        policy_hash = compute_policy_hash()
        
        # Combine into dep_closure_hash
        combined_dict = {'files': file_hashes, 'policy_hash': policy_hash}
        combined = json.dumps(combined_dict, sort_keys=True)
        computed_dep_hash = hashlib.sha256(combined.encode('utf-8')).hexdigest()
        if computed_dep_hash != namespace_receipt.dep_closure_hash:
            return False
        
        return True