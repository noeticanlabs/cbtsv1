import unittest
import tempfile
import os
import noetica_nsc_phase1.nsc as nsc

class TestBundleBitStability(unittest.TestCase):
    def test_bundle_bit_stability(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            srcdir = os.path.join(tmpdir, 'src')
            os.makedirs(srcdir)
            with open(os.path.join(srcdir, 'test.nsc'), 'w') as f:
                f.write('= ∇² u')
            bundle1 = os.path.join(tmpdir, 'bundle1.zip')
            bundle2 = os.path.join(tmpdir, 'bundle2.zip')
            nsc.create_bundle(srcdir, bundle1)
            nsc.create_bundle(srcdir, bundle2)
            with open(bundle1, 'rb') as f1, open(bundle2, 'rb') as f2:
                self.assertEqual(f1.read(), f2.read())