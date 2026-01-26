#!/usr/bin/env python3
"""Test NSC_GR policy integration."""

from noetica_nsc_phase1.nsc import compute_module_manifest

def test_gr_policy():
    # Test with GR glyphs
    test_source = "ℋ 𝓜 𝔊 𝔇 𝔅 𝔄 𝔯 𝕋"
    manifest = compute_module_manifest(test_source)
    print("Manifest keys:", list(manifest.keys()))
    print("Policy:", manifest['policy'])
    print("Module ID includes policy hash:", len(manifest['module_id'].split('-')) == 6)
    assert 'policy' in manifest, "Policy not in manifest"
    assert manifest['policy']['H_max'] == 1e-8, f"Wrong H_max: {manifest['policy']['H_max']}"
    print("NSC_GR policy test passed!")

if __name__ == "__main__":
    test_gr_policy()