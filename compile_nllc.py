#!/usr/bin/env python3
"""
Compile NLLC script to NIR and prepare for VM execution.
"""

import sys
import os
sys.path.append('src')

from nllc.parse import parse
from nllc.lower_nir import Lowerer
import json
import dataclasses

def compile_nllc(source_path):
    # Read source
    with open(source_path, 'r') as f:
        source = f.read()

    # Parse
    # Parse
    program = parse(source)

    # Lower to NIR
    lowerer = Lowerer(source_path)
    module = lowerer.lower_program(program)
    print("Number of blocks:", len(module.functions[0].blocks))

    # Serialize to JSON
    nir_json = json.dumps(dataclasses.asdict(module), indent=2)

    return nir_json, module

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python3 compile_nllc.py <source.nllc>")
        sys.exit(1)
    source_path = sys.argv[1]
    nir_json, module = compile_nllc(source_path)

    # Output the NIR
    print(nir_json)

    # Optionally, save to file
    with open("compiled_nir.json", "w") as f:
        f.write(nir_json)

    print("NIR compiled and saved to compiled_nir.json")