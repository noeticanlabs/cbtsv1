import unittest
import unittest.mock
import tempfile
import os
import noetica_nsc_phase1.nsc as nsc

class TestVerifyDetectsOpcodeTableChange(unittest.TestCase):
    def test_verify_detects_opcode_table_change(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            srcdir = os.path.join(tmpdir, 'src')
            os.makedirs(srcdir)
            with open(os.path.join(srcdir, 'test.nsc'), 'w') as f:
                f.write('= ∇² u')
            bundle = os.path.join(tmpdir, 'bundle.zip')
            nsc.create_bundle(srcdir, bundle)
            with unittest.mock.patch.object(nsc, 'GLYPH_TO_OPCODE', {**nsc.GLYPH_TO_OPCODE, '∇²': 99}):
                verify_dict, verified = nsc.verify_bundle(bundle)
                self.assertFalse(verified)