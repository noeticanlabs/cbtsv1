import unittest
import noetica_nsc_phase1.nsc as nsc

class TestModuleIdStableAcrossRuns(unittest.TestCase):
    def test_module_id_stable_across_runs(self):
        source = "= ∇² u"
        manifest1 = nsc.compute_module_manifest(source)
        manifest2 = nsc.compute_module_manifest(source)
        self.assertEqual(manifest1['module_id'], manifest2['module_id'])