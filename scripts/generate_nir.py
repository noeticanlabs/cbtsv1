#!/usr/bin/env python3
"""
Generate a simple NIR module for VM testing.
This creates a placeholder NIR module that can be loaded by the VM.
"""
import json
import os

def create_simple_nir_module():
    """Create a simple NIR module for testing."""
    
    # Define the NIR module structure
    module = {
        "schema": "nir_v0.1",
        "functions": [
            {
                "name": "gr_step",
                "params": [
                    {"name": "%dt", "type": "float64"},
                    {"name": "%t", "type": "float64"}
                ],
                "return_ty": {"kind": "float64"},
                "blocks": [
                    {
                        "name": "entry",
                        "instructions": [
                            {
                                "trace": {
                                    "file": "generated.nir",
                                    "span": {"start": 1, "end": 10},
                                    "ast_path": "gr_step"
                                },
                                "result": {"name": "%dt_copy", "type": "float64"},
                                "value": {"kind": "float64", "value": 0.001}
                            },
                            {
                                "trace": {
                                    "file": "generated.nir",
                                    "span": {"start": 11, "end": 20},
                                    "ast_path": "gr_step"
                                },
                                "result": {"name": "%result", "type": "float64"},
                                "value": {"kind": "float64", "value": 1.0}
                            },
                            {
                                "trace": {
                                    "file": "generated.nir",
                                    "span": {"start": 21, "end": 25},
                                    "ast_path": "gr_step"
                                },
                                "value": None,
                                "kind": "return"
                            }
                        ]
                    }
                ]
            },
            {
                "name": "gr_constraint_check",
                "params": [],
                "return_ty": {"kind": "float64"},
                "blocks": [
                    {
                        "name": "entry",
                        "instructions": [
                            {
                                "trace": {
                                    "file": "generated.nir",
                                    "span": {"start": 30, "end": 40},
                                    "ast_path": "gr_constraint_check"
                                },
                                "result": {"name": "%eps_H", "type": "float64"},
                                "value": {"kind": "float64", "value": 1e-10}
                            },
                            {
                                "trace": {
                                    "file": "generated.nir",
                                    "span": {"start": 41, "end": 45},
                                    "ast_path": "gr_constraint_check"
                                },
                                "value": None,
                                "kind": "return"
                            }
                        ]
                    }
                ]
            }
        ]
    }
    
    return module


def create_mms_nir_module():
    """Create a NIR module specifically for MMS test."""
    
    # More comprehensive NIR for MMS
    module = {
        "schema": "nir_v0.1",
        "metadata": {
            "version": "0.1.0",
            "generated_by": "mms_nir_generator",
            "purpose": "MMS convergence test VM"
        },
        "functions": [
            {
                "name": "mms_exact_fields",
                "params": [
                    {"name": "%N", "type": "int64"},
                    {"name": "%L", "type": "float64"},
                    {"name": "%t", "type": "float64"}
                ],
                "return_ty": {"kind": "object"},
                "blocks": [
                    {
                        "name": "entry",
                        "instructions": [
                            {
                                "trace": {
                                    "file": "mms.nir",
                                    "span": {"start": 1, "end": 50},
                                    "ast_path": "mms_exact_fields"
                                },
                                "result": {"name": "%gamma", "type": "array<float64>"},
                                "value": {"kind": "constant_array", "dims": [4, 4]}
                            },
                            {
                                "trace": {
                                    "file": "mms.nir",
                                    "span": {"start": 51, "end": 60},
                                    "ast_path": "mms_exact_fields"
                                },
                                "result": {"name": "%K", "type": "array<float64>"},
                                "value": {"kind": "constant_array", "dims": [4, 4]}
                            },
                            {
                                "trace": {
                                    "file": "mms.nir",
                                    "span": {"start": 61, "end": 65},
                                    "ast_path": "mms_exact_fields"
                                },
                                "value": None,
                                "kind": "return"
                            }
                        ]
                    }
                ]
            },
            {
                "name": "mms_rhs",
                "params": [
                    {"name": "%gamma", "type": "array<float64>"},
                    {"name": "%K", "type": "array<float64>"},
                    {"name": "%alpha", "type": "array<float64>"},
                    {"name": "%beta", "type": "array<float64>"}
                ],
                "return_ty": {"kind": "object"},
                "blocks": [
                    {
                        "name": "entry",
                        "instructions": [
                            {
                                "trace": {
                                    "file": "mms.nir",
                                    "span": {"start": 100, "end": 150},
                                    "ast_path": "mms_rhs"
                                },
                                "result": {"name": "%dt_gamma", "type": "array<float64>"},
                                "op": "mul",
                                "left": {"name": "%alpha", "type": "array<float64>"},
                                "right": {"name": "%K", "type": "array<float64>"}

                            },
                            {
                                "trace": {
                                    "file": "mms.nir",
                                    "span": {"start": 151, "end": 155},
                                    "ast_path": "mms_rhs"
                                },
                                "value": None,
                                "kind": "return"
                            }
                        ]
                    }
                ]
            }
        ]
    }
    
    return module


def main():
    """Generate NIR modules."""
    print("Generating NIR modules for VM testing...")
    
    # Create output directory
    output_dir = os.path.dirname(os.path.dirname(__file__))
    
    # Generate simple module
    simple_module = create_simple_nir_module()
    simple_path = os.path.join(output_dir, 'compiled_nir.json')
    
    with open(simple_path, 'w') as f:
        json.dump(simple_module, f, indent=2)
    
    print(f"Generated: {simple_path}")
    print(f"  Schema: {simple_module['schema']}")
    print(f"  Functions: {[f['name'] for f in simple_module['functions']]}")
    
    # Generate MMS-specific module
    mms_module = create_mms_nir_module()
    mms_path = os.path.join(output_dir, 'mms_nir.json')
    
    with open(mms_path, 'w') as f:
        json.dump(mms_module, f, indent=2)
    
    print(f"Generated: {mms_path}")
    print(f"  Schema: {mms_module['schema']}")
    print(f"  Functions: {[f['name'] for f in mms_module['functions']]}")
    
    print("\nTo enable VM, copy mms_nir.json to compiled_nir.json:")
    print(f"  cp {mms_path} {simple_path}")
    
    print("\nNote: The VM will be enabled but the NIR module is a placeholder.")
    print("For full VM functionality, proper NIR code generation is required.")


if __name__ == "__main__":
    main()
