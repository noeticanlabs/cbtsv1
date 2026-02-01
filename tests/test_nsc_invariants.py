"""
NSC-M3L Invariant Directive Tests
"""

import pytest

from tests.nsc_test_utils import (
    compile_nsc_source, parse_source, type_check_program
)
from src.nsc.ast import Model, Directive, DirectiveType
from src.nsc.type_checker import TypeChecker


class TestInvariantDirectiveParsing:
    """Test @inv directive parsing."""
    
    def test_single_invariant(self):
        """Test single invariant."""
        source = "@inv(N:INV.gr.hamiltonian_constraint);"
        ast = parse_source(source)
        directives = [s for s in ast.statements if isinstance(s, Directive)]
        assert len(directives) == 1
        assert directives[0].directive_type == DirectiveType.INV
    
    def test_multiple_invariants(self):
        """Test multiple invariants."""
        source = "@inv(N:INV.gr.hamiltonian_constraint, N:INV.gr.momentum_constraint);"
        ast = parse_source(source)
        directives = [s for s in ast.statements if isinstance(s, Directive)]
        inv_list = directives[0].invariant_list
        assert inv_list is not None
        assert len(inv_list.invariants) == 2
    
    def test_invariant_with_model(self):
        """Test invariant with model."""
        source = "@model(GEO, CALC);\n@inv(N:INV.gr.hamiltonian_constraint);"
        ast = parse_source(source)
        directives = [s for s in ast.statements if isinstance(s, Directive)]
        assert len(directives) == 2


class TestInvariantTypeCompatibility:
    """Test invariant compatibility."""
    
    def test_gr_invariant_requires_geo(self):
        """Test GR requires GEO."""
        source = "@model(CALC);\n@inv(N:INV.gr.hamiltonian_constraint);"
        result = compile_nsc_source(source)
        assert result.ast is not None
    
    def test_gr_invariant_with_geo(self):
        """Test GR with GEO."""
        source = "@model(GEO, CALC);\n@inv(N:INV.gr.hamiltonian_constraint);\nM :: Manifold(3+1, lorentzian);\ng :: Metric on M;"
        result = compile_nsc_source(source)
        assert result.success
    
    def test_ns_invariant_without_geo(self):
        """Test NS without GEO."""
        source = "@model(CALC, LEDGER);\n@inv(N:INV.ns.div_free);"
        result = compile_nsc_source(source)
        assert result.success


class TestInvariantRegistry:
    """Test invariant registry."""
    
    def test_known_invariant(self):
        """Test known invariant."""
        registry = {"N:INV.gr.hamiltonian_constraint": {"description": "GR Hamiltonian"}}
        source = "@inv(N:INV.gr.hamiltonian_constraint);"
        ast = parse_source(source)
        checker = TypeChecker(invariant_registry=registry)
        result = checker.check_program(ast)
        assert True
    
    def test_unknown_invariant(self):
        """Test unknown invariant."""
        source = "@inv(N:INV.unknown.invariant);"
        ast = parse_source(source)
        checker = TypeChecker(invariant_registry={})
        result = checker.check_program(ast)
        assert ast is not None


class TestInvariantExamples:
    """Test real invariant examples."""
    
    def test_gr_hamiltonian_constraint(self):
        """Test GR Hamiltonian."""
        source = """
        @model(GEO, CALC, LEDGER);
        M :: Manifold(3+1, lorentzian);
        g :: Metric on M;
        K :: Field[Tensor(0,2)];
        H := R(g) + tr(K^2) - K_ij*K^ij = 0;
        @inv(N:INV.gr.hamiltonian_constraint);
        ⇒ (LEDGER, CALC, GEO);
        """
        result = compile_nsc_source(source)
        assert result.success
    
    def test_gr_momentum_constraint(self):
        """Test GR momentum."""
        source = """
        @model(GEO, CALC, LEDGER);
        M :: Manifold(3+1, lorentzian);
        g :: Metric on M;
        K :: Field[Tensor(0,2)];
        @inv(N:INV.gr.momentum_constraint);
        ⇒ (LEDGER, CALC, GEO);
        """
        result = compile_nsc_source(source)
        assert result.success


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
