import unittest
import os
import json
import tempfile
import time
from unittest.mock import patch
from noetica_nsc_phase1 import nsc_cli

class TestCacheGcDeterministic(unittest.TestCase):
    def test_cache_gc_deterministic(self):
        with tempfile.TemporaryDirectory() as tmp_cache:
            with patch('noetica_nsc_phase1.nsc_cache.cache_root', return_value=tmp_cache):
                fake_ids = ['fake1', 'fake2', 'fake3', 'fake4']
                for i, fid in enumerate(fake_ids):
                    cache_d = os.path.join(tmp_cache, fid)
                    os.makedirs(cache_d)
                    manifest_path = os.path.join(cache_d, 'manifest.json')
                    with open(manifest_path, 'w') as f:
                        json.dump({'module_id': fid}, f)
                    # Set mtime, fake1 oldest, fake4 newest
                    mtime = time.time() - (len(fake_ids) - i) * 10  # fake1 40s ago, fake4 10s ago
                    os.utime(manifest_path, (mtime, mtime))

                # Run gc with keep=2
                nsc_cli.do_cache_gc(keep=2)

                # Check that only fake3 and fake4 remain
                remaining = []
                for d in os.listdir(tmp_cache):
                    cache_d = os.path.join(tmp_cache, d)
                    if os.path.isdir(cache_d):
                        manifest_path = os.path.join(cache_d, "manifest.json")
                        if os.path.exists(manifest_path):
                            remaining.append(d)
                remaining.sort()  # should be fake3, fake4
                self.assertEqual(remaining, ['fake3', 'fake4'])

if __name__ == '__main__':
    unittest.main()