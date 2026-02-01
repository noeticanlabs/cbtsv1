"""
NSC-M3L Performance Benchmark Tests

Tests compilation performance and scalability.
"""

import pytest
import time

from tests.nsc_test_utils import (
    compile_nsc_source, parse_source, type_check_program,
    CompilationResult
)
from src.nsc.ast import Model


class TestCompilationSpeedSimple:
    """Test compilation speed for simple programs."""
    
    def test_simple_scalar_compilation_speed(self):
        """Test compilation time for simple scalar program."""
        source = "x :: Scalar;"
        
        start = time.time()
        for _ in range(100):
            compile_nsc_source(source)
        elapsed = time.time() - start
        
        assert elapsed < 5.0, f"100 compilations took {elapsed:.2f}s"
    
    def test_simple_field_compilation_speed(self):
        """Test compilation time for simple field program."""
        source = "@model(CALC);\nu :: Field[Scalar];"
        
        start = time.time()
        for _ in range(50):
            compile_nsc_source(source)
        elapsed = time.time() - start
        
        assert elapsed < 5.0, f"50 compilations took {elapsed:.2f}s"
    
    def test_simple_equation_compilation_speed(self):
        """Test compilation time for simple equation."""
        source = "@model(CALC);\nu :: Field[Scalar];\nf :: Field[Scalar];\nΔ(u) = f;"
        
        start = time.time()
        for _ in range(30):
            compile_nsc_source(source)
        elapsed = time.time() - start
        
        assert elapsed < 5.0, f"30 compilations took {elapsed:.2f}s"


class TestCompilationSpeedComplex:
    """Test compilation speed for complex programs."""
    
    def test_gr_program_compilation_speed(self):
        """Test compilation time for GR program."""
        source = """
        @model(GEO, CALC, LEDGER);
        M :: Manifold(3+1, lorentzian);
        g :: Metric on M;
        K :: Field[Tensor(0,2)];
        H := R(g) + tr(K^2) - K_ij*K^ij = 0;
        @inv(N:INV.gr.hamiltonian_constraint);
        ⇒ (LEDGER, CALC, GEO);
        """
        
        start = time.time()
        compile_nsc_source(source)
        elapsed = time.time() - start
        
        assert elapsed < 2.0, f"GR compilation took {elapsed:.2f}s"
    
    def test_disc_lowering_speed(self):
        """Test DISC lowering speed."""
        source = """
        @model(CALC, DISC);
        u :: Field[Scalar];
        f :: Field[Scalar];
        Δ(u) = f;
        ⇒ (DISC);
        """
        
        start = time.time()
        for _ in range(20):
            compile_nsc_source(source, target_models={Model.DISC})
        elapsed = time.time() - start
        
        assert elapsed < 10.0, f"20 DISC compilations took {elapsed:.2f}s"


class TestParsingSpeed:
    """Test parsing speed."""
    
    def test_simple_parse_speed(self):
        """Test simple parsing speed."""
        source = "x :: Scalar;"
        
        start = time.time()
        for _ in range(500):
            parse_source(source)
        elapsed = time.time() - start
        
        assert elapsed < 2.0, f"500 parses took {elapsed:.2f}s"
    
    def test_complex_parse_speed(self):
        """Test complex parsing speed."""
        source = """
        @model(GEO, CALC, LEDGER, EXEC);
        M :: Manifold(3+1, lorentzian);
        g :: Metric on M;
        K :: Field[Tensor(0,2)];
        T :: Field[Tensor(0,2)];
        H := R(g) + tr(K^2) - K_ij*K^ij = 0;
        M_i := ∇^j(K_ij - K g_ij) = 0;
        @inv(N:INV.gr.hamiltonian_constraint, N:INV.gr.momentum_constraint);
        ⇒ (LEDGER, CALC, GEO, EXEC);
        """
        
        start = time.time()
        for _ in range(100):
            parse_source(source)
        elapsed = time.time() - start
        
        assert elapsed < 5.0, f"100 parses took {elapsed:.2f}s"


class TestTypeCheckingSpeed:
    """Test type checking speed."""
    
    def test_simple_type_check_speed(self):
        """Test simple type checking speed."""
        source = "x :: Scalar;\ny :: Scalar;\nz :: Scalar;\nw = x + y + z;"
        
        start = time.time()
        for _ in range(100):
            type_check_program(parse_source(source))
        elapsed = time.time() - start
        
        assert elapsed < 5.0, f"100 type checks took {elapsed:.2f}s"
    
    def test_complex_type_check_speed(self):
        """Test complex type checking speed."""
        source = """
        @model(GEO, CALC);
        u :: Field[Scalar];
        v :: Field[Vector];
        T :: Field[Tensor(0,2)];
        Eq1 := Δ(u) = f;
        Eq2 := div(v) = 0;
        Eq3 := curl(v) = w;
        """
        
        start = time.time()
        for _ in range(50):
            type_check_program(parse_source(source))
        elapsed = time.time() - start
        
        assert elapsed < 5.0, f"50 type checks took {elapsed:.2f}s"


class TestScalability:
    """Test scalability with program size."""
    
    def test_many_declarations(self):
        """Test with many declarations."""
        decls = "\n".join(f"x{i} :: Scalar;" for i in range(100))
        source = f"@model(CALC);\n{decls}"
        
        start = time.time()
        result = compile_nsc_source(source)
        elapsed = time.time() - start
        
        assert result.success
        assert elapsed < 3.0, f"100 declarations took {elapsed:.2f}s"
    
    def test_many_equations(self):
        """Test with many equations."""
        eqs = "\n".join(f"Eq{i} := Δ(u{i}) = f{i};" for i in range(20))
        decls = "\n".join(f"u{i} :: Field[Scalar]; f{i} :: Field[Scalar];" for i in range(20))
        source = f"@model(CALC);\n{decls}\n{eqs}"
        
        start = time.time()
        result = compile_nsc_source(source)
        elapsed = time.time() - start
        
        assert result.success
        assert elapsed < 3.0, f"20 equations took {elapsed:.2f}s"


class TestMemoryStability:
    """Test memory stability."""
    
    def test_no_memory_leak_simple(self):
        """Test no memory leak with simple repeated compilations."""
        source = "x :: Scalar;"
        for _ in range(100):
            compile_nsc_source(source)
        assert True
    
    def test_no_memory_leak_complex(self):
        """Test no memory leak with complex repeated compilations."""
        source = """
        @model(GEO, CALC, LEDGER, EXEC);
        M :: Manifold(3+1, lorentzian);
        g :: Metric on M;
        K :: Field[Tensor(0,2)];
        H := R(g) + tr(K^2) - K_ij*K^ij = 0;
        ⇒ (LEDGER, CALC, GEO, EXEC);
        """
        for _ in range(20):
            compile_nsc_source(source)
        assert True


class TestThroughput:
    """Test overall throughput."""
    
    def test_compilations_per_second_simple(self):
        """Test compilations per second for simple programs."""
        source = "x :: Scalar;"
        
        start = time.time()
        count = 0
        elapsed = 0.0
        while elapsed < 1.0:
            compile_nsc_source(source)
            count += 1
            elapsed = time.time() - start
        
        assert count >= 10, f"Only {count} compilations in 1 second"
    
    def test_compilations_per_second_complex(self):
        """Test compilations per second for complex programs."""
        source = "@model(GEO, CALC);\nu :: Field[Scalar];\nΔ(u) = f;"
        
        start = time.time()
        count = 0
        elapsed = 0.0
        while elapsed < 1.0:
            compile_nsc_source(source)
            count += 1
            elapsed = time.time() - start
        
        assert count >= 5, f"Only {count} compilations in 1 second"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
