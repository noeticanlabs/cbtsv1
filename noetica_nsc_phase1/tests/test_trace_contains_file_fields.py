import unittest
import tempfile
import os
import json
import noetica_nsc_phase1.nsc_module as nsc_module

class TestTraceContainsFileFields(unittest.TestCase):
    def test_trace_contains_file_fields(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create manifest
            manifest_path = os.path.join(tmpdir, 'nsc.module.json')
            manifest = {
                'sources': ['a.nsc', 'b.nsc'],
                'imports': []
            }
            with open(manifest_path, 'w') as f:
                json.dump(manifest, f)
            
            # Create src files
            with open(os.path.join(tmpdir, 'a.nsc'), 'w') as f:
                f.write('= ∇ u')
            with open(os.path.join(tmpdir, 'b.nsc'), 'w') as f:
                f.write('= ∇² u')
            
            # Compile
            loaded_manifest = nsc_module.load_module_manifest(tmpdir)
            artifact = nsc_module.compile_module(tmpdir, loaded_manifest)
            
            # Check trace
            self.assertEqual(len(artifact.module_trace), len(artifact.module_bytecode.opcodes))
            for entry in artifact.module_trace:
                if entry.path.startswith('files.'):
                    self.assertIsNotNone(entry.file)
                self.assertIsNotNone(entry.file_sentence)
                self.assertIsNotNone(entry.module_sentence)