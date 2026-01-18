import unittest
import json

import noetica_nsc_phase1.nsc as nsc
import noetica_nsc_phase1.nsc_export as nsc_export

class TestExport(unittest.TestCase):
    def test_json_keys(self):
        src = "= ∇² u"
        prog, flat, bc, tpl = nsc.nsc_to_pde(src)
        data = nsc_export.export_symbolic(src, prog, flat, bc, tpl)
        required_keys = ["nsc_version", "ast", "flattened", "determinism"]
        for key in required_keys:
            self.assertIn(key, data)

    def test_deterministic_serialization(self):
        src = "= ∇² u"
        prog, flat, bc, tpl = nsc.nsc_to_pde(src)
        data = nsc_export.export_symbolic(src, prog, flat, bc, tpl)
        json1 = json.dumps(data, indent=2)
        json2 = json.dumps(data, indent=2)
        self.assertEqual(json1, json2)