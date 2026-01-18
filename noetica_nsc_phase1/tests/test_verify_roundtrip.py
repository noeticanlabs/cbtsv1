import unittest
import tempfile
import os
import noetica_nsc_phase1.nsc as nsc

class TestVerifyRoundtrip(unittest.TestCase):
    def test_verify_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            srcdir = os.path.join(tmpdir, 'src')
            os.makedirs(srcdir)
            with open(os.path.join(srcdir, 'test.nsc'), 'w') as f:
                f.write('= ∇² u')
            bundle = os.path.join(tmpdir, 'bundle.zip')
            nsc.create_bundle(srcdir, bundle)
            verify_dict, verified = nsc.verify_bundle(bundle)
            self.assertTrue(verified)