#!/usr/bin/env python3
"""
NLLC (Noetica Language for Logical Computing) CLI

Usage:
    nllc compile <input.nllc> [-o <output.nir>] [--verbose]
    nllc typecheck <input.nllc> [--json]
    nllc run <input.nllc> [--trace]
    nllc --version
    nllc --help

Options:
    -o, --output FILE    Output NIR file path
    -v, --verbose        Verbose output
    --json               JSON output for type checking
    --trace              Enable VM trace mode
"""

import argparse
import sys
import json
from pathlib import Path

from .lex import Lexer
from .parse import Parser
from .lower_nir import Lowerer
from .type_checker import TypeChecker
from .vm import VM


def cmd_compile(args):
    """Compile NLLC source to NIR."""
    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path.with_suffix('.nir')
    
    # Read source
    with open(input_path, 'r') as f:
        source = f.read()
    
    if args.verbose:
        print(f"Compiling: {input_path}")
        print(f"Output: {output_path}")
    
    # Lex
    lexer = Lexer(source)
    tokens = lexer.tokenize()
    
    if args.verbose:
        print(f"Tokens: {len(tokens)}")
    
    # Parse
    parser = Parser(tokens)
    ast = parser.parse_program()
    
    if args.verbose:
        print(f"AST nodes: {len(ast.statements)}")
    
    # Lower to NIR
    lowerer = Lowerer(str(input_path))
    module = lowerer.lower_program(ast)
    
    if args.verbose:
        print(f"NIR functions: {len(module.functions)}")
    
    # Type check
    type_checker = TypeChecker()
    result = type_checker.check(module)
    
    if result.errors:
        print("Type errors:")
        for error in result.errors:
            print(f"  {error}")
        return 1
    
    if args.verbose:
        print(f"Type check passed: {type_checker.type_count if hasattr(type_checker, 'type_count') else 'OK'}")
    
    # Write NIR as JSON
    output = {
        "functions": [
            {
                "name": f.name,
                "params": [{"name": p.name, "type": str(p.ty)} for p in f.params],
                "return_type": str(f.return_ty),
                "blocks": [
                    {
                        "name": b.name,
                        "instructions": [
                            _inst_to_dict(i) for i in b.instructions
                        ]
                    }
                    for b in f.blocks
                ]
            }
            for f in module.functions
        ]
    }
    
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    if args.verbose:
        total_insts = sum(len(f.blocks) * 4 for f in module.functions)  # approx
        print(f"NIR written to {output_path}")
    
    print(f"Compiled: {input_path} → {output_path}")
    return 0


def _inst_to_dict(inst):
    """Convert instruction to dict for JSON serialization."""
    if hasattr(inst, 'result') and inst.result:
        result = {"name": inst.result.name, "type": str(inst.result.ty)}
    else:
        result = None
    
    if hasattr(inst, 'trace') and inst.trace:
        trace = {"file": inst.trace.file, "span": str(inst.trace.span)}
    else:
        trace = None
    
    base = {"result": result, "trace": trace}
    
    if hasattr(inst, 'value') and inst.value is not None:
        base["value"] = inst.value if isinstance(inst.value, (int, float, bool, str)) else str(inst.value)
    elif hasattr(inst, 'left'):
        base["left"] = inst.left.name
        base["op"] = inst.op
        base["right"] = inst.right.name
    elif hasattr(inst, 'func'):
        base["func"] = inst.func
        base["args"] = [a.name for a in inst.args]
    elif hasattr(inst, 'ptr'):
        base["ptr"] = inst.ptr.name
        if inst.index:
            base["index"] = inst.index.name
    elif hasattr(inst, 'cond') and inst.cond:
        base["cond"] = inst.cond.name
        base["true_block"] = inst.true_block
        base["false_block"] = inst.false_block
    elif hasattr(inst, 'value') and inst.value is None:
        pass  # void return
    
    return {inst.__class__.__name__: base}


def cmd_typecheck(args):
    """Type check NLLC source."""
    input_path = Path(args.input)
    
    with open(input_path, 'r') as f:
        source = f.read()
    
    lexer = Lexer(source)
    tokens = lexer.tokenize()
    
    parser = Parser(tokens)
    ast = parser.parse_program()
    
    # Lower to NIR
    lowerer = Lowerer(str(input_path))
    module = lowerer.lower_program(ast)
    
    # Type check
    type_checker = TypeChecker()
    result = type_checker.check(module)
    
    if args.json:
        result_out = {
            "file": str(input_path),
            "valid": result.success,
            "functions": len(module.functions),
            "errors": [str(e) for e in result.errors]
        }
        print(json.dumps(result_out, indent=2))
    else:
        if not result.success:
            print("Type errors:")
            for error in result.errors:
                print(f"  {error}")
            return 1
        else:
            print(f"Type check passed ({len(module.functions)} functions)")
            return 0


def cmd_run(args):
    """Run NLLC source in VM."""
    input_path = Path(args.input)
    
    with open(input_path, 'r') as f:
        source = f.read()
    
    lexer = Lexer(source)
    tokens = lexer.tokenize()
    
    parser = Parser(tokens)
    ast = parser.parse_program()
    
    # Lower to NIR
    lowerer = Lowerer(str(input_path))
    module = lowerer.lower_program(ast)
    
    # Type check
    type_checker = TypeChecker()
    result = type_checker.check(module)
    
    if not result.success:
        print("Type errors:")
        for error in result.errors:
            print(f"  {error}")
        return 1
    
    # Run VM
    vm = VM(trace=args.trace)
    result = vm.run(module)
    
    print(f"VM result: {result}")
    return 0


def main():
    parser = argparse.ArgumentParser(description="NLLC Compiler and VM")
    parser.add_argument('--version', action='version', version='NLLC 2.0')
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Compile
    compile_parser = subparsers.add_parser('compile', help='Compile NLLC to NIR')
    compile_parser.add_argument('input', help='Input NLLC file')
    compile_parser.add_argument('-o', '--output', help='Output NIR file')
    compile_parser.add_argument('-v', '--verbose', action='store_true')
    
    # Typecheck
    type_parser = subparsers.add_parser('typecheck', help='Type check NLLC source')
    type_parser.add_argument('input', help='Input NLLC file')
    type_parser.add_argument('--json', action='store_true', help='JSON output')
    
    # Run
    run_parser = subparsers.add_parser('run', help='Run NLLC source')
    run_parser.add_argument('input', help='Input NLLC file')
    run_parser.add_argument('--trace', action='store_true', help='Enable trace')
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return 1
    
    commands = {
        'compile': cmd_compile,
        'typecheck': cmd_typecheck,
        'run': cmd_run
    }
    
    return commands[args.command](args)


if __name__ == '__main__':
    sys.exit(main())
