import unittest
import tempfile
import os
import json
import noetica_nsc_phase1.nsc_module as nsc_module

class TestModuleIdChangesOnImportChange(unittest.TestCase):
    def test_module_id_changes_when_import_changes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create manifest
            manifest_path = os.path.join(tmpdir, 'nsc.module.json')
            manifest = {
                'sources': ['a.nsc'],
                'imports': ['import1']
            }
            with open(manifest_path, 'w') as f:
                json.dump(manifest, f)
            
            # Create src file
            with open(os.path.join(tmpdir, 'a.nsc'), 'w') as f:
                f.write('= ∇ u')
            
            # Build first
            loaded_manifest = nsc_module.load_module_manifest(tmpdir)
            artifact1 = nsc_module.compile_module(tmpdir, loaded_manifest)
            id1 = artifact1.module_id
            
            # Change import in manifest
            manifest['imports'] = ['import2']
            with open(manifest_path, 'w') as f:
                json.dump(manifest, f)
            
            # Reload manifest (since sorted)
            loaded_manifest2 = nsc_module.load_module_manifest(tmpdir)
            artifact2 = nsc_module.compile_module(tmpdir, loaded_manifest2)
            id2 = artifact2.module_id
            
            self.assertNotEqual(id1, id2)