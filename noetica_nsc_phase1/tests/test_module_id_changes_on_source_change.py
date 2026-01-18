import unittest
import noetica_nsc_phase1.nsc as nsc

class TestModuleIdChangesOnSourceChange(unittest.TestCase):
    def test_module_id_changes_on_source_change(self):
        source1 = "φ⊕"
        source2 = "φ⊖"
        manifest1 = nsc.compute_module_manifest(source1)
        manifest2 = nsc.compute_module_manifest(source2)
        self.assertNotEqual(manifest1['module_id'], manifest2['module_id'])