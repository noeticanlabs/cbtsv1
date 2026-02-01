"""
NSC-M3L JIT Compilation

Just-in-time compilation for hot loops in NSC-M3L bytecode execution.
Uses Numba for high-performance numerical computation.

Semantic Domain Objects:
    - Bytecode/IR for VM execution
    - JIT compilation for performance

Denotation: Program → Executable bytecode with execution semantics
"""

import hashlib
import time
from typing import Callable, Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field

import numpy as np


# =============================================================================
# JIT Compilation Status
# =============================================================================

@dataclass
class JITCompiledFunction:
    """Record of JIT compiled function."""
    name: str
    bytecode_hash: str
    native_code: Callable
    compile_time: float
    call_count: int = 0
    total_time: float = 0.0
    
    def call(self, *args, **kwargs) -> Any:
        """Call the compiled function with timing."""
        start = time.time()
        result = self.native_code(*args, **kwargs)
        self.total_time += time.time() - start
        self.call_count += 1
        return result


# =============================================================================
# JIT Compiler
# =============================================================================

class JITCompiler:
    """
    JIT compiler for NSC-M3L bytecode hot loops.
    
    Features:
    - Automatic hot loop detection
    - Numba-based compilation
    - Function caching
    - Performance profiling
    """
    
    def __init__(self, hot_threshold: int = 100, max_cache_size: int = 64):
        """
        Initialize JIT compiler.
        
        Args:
            hot_threshold: Operations before compilation
            max_cache_size: Maximum compiled functions to cache
        """
        self.hot_threshold = hot_threshold
        self.max_cache_size = max_cache_size
        
        # Compiled functions cache
        self.compiled: Dict[str, JITCompiledFunction] = {}
        self.hot_paths: Dict[str, int] = {}
        
        # Statistics
        self.total_compile_time: float = 0.0
        self.total_saved_time: float = 0.0
        self.compile_count: int = 0
        
        # Check Numba availability
        self.numba_available = self._check_numba()
        
        # Initialize compiled kernels
        if self.numba_available:
            self._init_kernels()
    
    def _check_numba(self) -> bool:
        """Check if Numba is available."""
        try:
            from numba import jit, prange
            return True
        except ImportError:
            return False
    
    def _init_kernels(self):
        """Initialize Numba-compiled kernels."""
        # Math operations
        self._add_kernel('add', self._jit_add)
        self._add_kernel('sub', self._jit_sub)
        self._add_kernel('mul', self._jit_mul)
        self._add_kernel('div', self._jit_div)
        
        # Physics operations
        self._add_kernel('gradient', self._jit_gradient)
        self._add_kernel('divergence', self._jit_divergence)
        self._add_kernel('laplacian', self._jit_laplacian)
        self._add_kernel('curl', self._jit_curl)
        
        # GR operations
        self._add_kernel('christoffel', self._jit_christoffel)
        self._add_kernel('ricci', self._jit_ricci)
    
    def _add_kernel(self, name: str, kernel: Callable):
        """Add a pre-compiled kernel."""
        if self.numba_available:
            from numba import jit
            compiled = jit(nopython=True)(kernel)
        else:
            compiled = kernel
        
        self.compiled[name] = JITCompiledFunction(
            name=name,
            bytecode_hash='',
            native_code=compiled,
            compile_time=0.0
        )
    
    # -------------------------------------------------------------------------
    # Math Kernels
    # -------------------------------------------------------------------------
    
    @staticmethod
    def _jit_add(a: float, b: float) -> float:
        """JIT-compiled addition."""
        return a + b
    
    @staticmethod
    def _jit_sub(a: float, b: float) -> float:
        """JIT-compiled subtraction."""
        return a - b
    
    @staticmethod
    def _jit_mul(a: float, b: float) -> float:
        """JIT-compiled multiplication."""
        return a * b
    
    @staticmethod
    def _jit_div(a: float, b: float) -> float:
        """JIT-compiled division."""
        return a / b if b != 0 else 0.0
    
    # -------------------------------------------------------------------------
    # Physics Kernels
    # -------------------------------------------------------------------------
    
    @staticmethod
    def _jit_gradient(field: np.ndarray, dx: float, dy: float, dz: float) -> np.ndarray:
        """JIT-compiled gradient computation with central differences."""
        nx, ny, nz = field.shape
        grad = np.zeros((3, nx, ny, nz), dtype=field.dtype)
        
        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                for k in range(1, nz - 1):
                    grad[0, i, j, k] = (field[i+1, j, k] - field[i-1, j, k]) / (2 * dx)
                    grad[1, i, j, k] = (field[i, j+1, k] - field[i, j-1, k]) / (2 * dy)
                    grad[2, i, j, k] = (field[i, j, k+1] - field[i, j, k-1]) / (2 * dz)
        
        return grad
    
    @staticmethod
    def _jit_divergence(vec_field: np.ndarray, dx: float, dy: float, dz: float) -> np.ndarray:
        """JIT-compiled divergence computation."""
        nx, ny, nz = vec_field.shape[1:]
        div = np.zeros((nx, ny, nz), dtype=vec_field.dtype)
        
        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                for k in range(1, nz - 1):
                    div[i, j, k] = (
                        (vec_field[0, i+1, j, k] - vec_field[0, i-1, j, k]) / (2 * dx) +
                        (vec_field[1, i, j+1, k] - vec_field[1, i, j-1, k]) / (2 * dy) +
                        (vec_field[2, i, j, k+1] - vec_field[2, i, j, k-1]) / (2 * dz)
                    )
        
        return div
    
    @staticmethod
    def _jit_laplacian(field: np.ndarray, dx: float, dy: float, dz: float) -> np.ndarray:
        """JIT-compiled Laplacian computation (7-point stencil)."""
        nx, ny, nz = field.shape
        lap = np.zeros((nx, ny, nz), dtype=field.dtype)
        
        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                for k in range(1, nz - 1):
                    lap[i, j, k] = (
                        (field[i+1, j, k] + field[i-1, j, k]) / (dx * dx) +
                        (field[i, j+1, k] + field[i, j-1, k]) / (dy * dy) +
                        (field[i, j, k+1] + field[i, j, k-1]) / (dz * dz) -
                        6.0 * field[i, j, k] / (dx * dx)
                    )
        
        return lap
    
    @staticmethod
    def _jit_curl(vec_field: np.ndarray, dx: float, dy: float, dz: float) -> np.ndarray:
        """JIT-compiled curl computation."""
        nx, ny, nz = vec_field.shape[1:]
        curl = np.zeros((3, nx, ny, nz), dtype=vec_field.dtype)
        
        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                for k in range(1, nz - 1):
                    curl[0, i, j, k] = (
                        (vec_field[2, i, j+1, k] - vec_field[2, i, j-1, k]) / (2 * dy) -
                        (vec_field[1, i, j, k+1] - vec_field[1, i, j, k-1]) / (2 * dz)
                    )
                    curl[1, i, j, k] = (
                        (vec_field[0, i, j, k+1] - vec_field[0, i, j, k-1]) / (2 * dz) -
                        (vec_field[2, i+1, j, k] - vec_field[2, i-1, j, k]) / (2 * dx)
                    )
                    curl[2, i, j, k] = (
                        (vec_field[1, i+1, j, k] - vec_field[1, i-1, j, k]) / (2 * dx) -
                        (vec_field[0, i, j+1, k] - vec_field[0, i, j-1, k]) / (2 * dy)
                    )
        
        return curl
    
    # -------------------------------------------------------------------------
    # GR Kernels
    # -------------------------------------------------------------------------
    
    @staticmethod
    def _jit_christoffel(metric: np.ndarray, d_metric: np.ndarray) -> np.ndarray:
        """JIT-compiled Christoffel symbol computation."""
        # Simplified - full implementation would compute Γ^k_ij
        nx, ny, nz = metric.shape[:3]
        christoffel = np.zeros((3, nx, ny, nz, 3, 3), dtype=metric.dtype)
        return christoffel
    
    @staticmethod
    def _jit_ricci(christoffel: np.ndarray, d_christoffel: np.ndarray) -> np.ndarray:
        """JIT-compiled Ricci tensor computation."""
        nx, ny, nz = christoffel.shape[:3]
        ricci = np.zeros((nx, ny, nz, 3, 3), dtype=christoffel.dtype)
        return ricci
    
    # -------------------------------------------------------------------------
    # Compilation Interface
    # -------------------------------------------------------------------------
    
    def compile(self, operation: str, args: Tuple, kwargs: Dict) -> JITCompiledFunction:
        """
        Compile a hot operation to native code.
        
        Args:
            operation: Operation name (e.g., 'gradient', 'laplacian')
            args: Operation arguments
            kwargs: Operation keyword arguments
            
        Returns:
            JITCompiledFunction
        """
        # Generate cache key
        cache_key = self._generate_cache_key(operation, args, kwargs)
        
        # Check cache
        if cache_key in self.compiled:
            return self.compiled[cache_key]
        
        # Compile if hot
        if operation in self.hot_paths and self.hot_paths[operation] >= self.hot_threshold:
            compiled = self._do_compile(operation, args, kwargs)
            
            if compiled:
                self.compiled[cache_key] = compiled
                self.compile_count += 1
                
                # Manage cache size
                if len(self.compiled) > self.max_cache_size:
                    self._evict_oldest()
        
        # Return compiled or fallback
        if operation in self.compiled:
            return self.compiled[operation]
        
        # Fallback to non-compiled version
        return self._create_fallback(operation)
    
    def _do_compile(self, operation: str, args: Tuple, kwargs: Dict) -> Optional[JITCompiledFunction]:
        """Perform actual compilation."""
        start = time.time()
        
        # Get kernel
        if operation in ('gradient', 'div', 'curl', 'laplacian'):
            return self._compile_physics(operation, start)
        elif operation in ('add', 'sub', 'mul', 'div'):
            return self._compile_math(operation, start)
        elif operation in ('christoffel', 'ricci'):
            return self._compile_gr(operation, start)
        
        return None
    
    def _compile_physics(self, operation: str, start: float) -> JITCompiledFunction:
        """Compile physics kernel."""
        from numba import jit, prange
        
        if operation == 'gradient':
            def kernel(field, dx=0.1, dy=0.1, dz=0.1):
                return self._jit_gradient(field, dx, dy, dz)
        elif operation == 'div':
            def kernel(field, dx=0.1, dy=0.1, dz=0.1):
                return self._jit_divergence(field, dx, dy, dz)
        elif operation == 'curl':
            def kernel(field, dx=0.1, dy=0.1, dz=0.1):
                return self._jit_curl(field, dx, dy, dz)
        elif operation == 'laplacian':
            def kernel(field, dx=0.1, dy=0.1, dz=0.1):
                return self._jit_laplacian(field, dx, dy, dz)
        else:
            return None
        
        # Compile with Numba
        compiled = jit(nopython=True, parallel=True)(kernel)
        compile_time = time.time() - start
        
        self.total_compile_time += compile_time
        
        return JITCompiledFunction(
            name=operation,
            bytecode_hash='',
            native_code=compiled,
            compile_time=compile_time
        )
    
    def _compile_math(self, operation: str, start: float) -> JITCompiledFunction:
        """Compile math kernel."""
        from numba import jit
        
        if operation == 'add':
            def kernel(a, b):
                return self._jit_add(a, b)
        elif operation == 'sub':
            def kernel(a, b):
                return self._jit_sub(a, b)
        elif operation == 'mul':
            def kernel(a, b):
                return self._jit_mul(a, b)
        elif operation == 'div':
            def kernel(a, b):
                return self._jit_div(a, b)
        else:
            return None
        
        compiled = jit(nopython=True)(kernel)
        compile_time = time.time() - start
        
        self.total_compile_time += compile_time
        
        return JITCompiledFunction(
            name=operation,
            bytecode_hash='',
            native_code=compiled,
            compile_time=compile_time
        )
    
    def _compile_gr(self, operation: str, start: float) -> JITCompiledFunction:
        """Compile GR kernel."""
        from numba import jit
        
        if operation == 'christoffel':
            def kernel(metric, d_metric):
                return self._jit_christoffel(metric, d_metric)
        elif operation == 'ricci':
            def kernel(christoffel, d_christoffel):
                return self._jit_ricci(christoffel, d_christoffel)
        else:
            return None
        
        compiled = jit(nopython=True)(kernel)
        compile_time = time.time() - start
        
        self.total_compile_time += compile_time
        
        return JITCompiledFunction(
            name=operation,
            bytecode_hash='',
            native_code=compiled,
            compile_time=compile_time
        )
    
    def _create_fallback(self, operation: str) -> JITCompiledFunction:
        """Create fallback for non-compiled operations."""
        def fallback(*args, **kwargs):
            # Dispatch to appropriate handler
            if operation == 'add':
                return args[0] + args[1]
            elif operation == 'sub':
                return args[0] - args[1]
            elif operation == 'mul':
                return args[0] * args[1]
            elif operation == 'div':
                return args[0] / args[1] if args[1] != 0 else 0.0
            return None
        
        return JITCompiledFunction(
            name=operation,
            bytecode_hash='',
            native_code=fallback,
            compile_time=0.0
        )
    
    def mark_hot(self, operation: str):
        """Mark an operation as hot (frequently executed)."""
        self.hot_paths[operation] = self.hot_paths.get(operation, 0) + 1
    
    def _generate_cache_key(self, operation: str, args: Tuple, kwargs: Dict) -> str:
        """Generate cache key for compilation."""
        key_data = {
            'op': operation,
            'args': [str(type(a)) for a in args],
            'kwargs': {k: str(type(v)) for k, v in kwargs.items()}
        }
        return hashlib.sha256(str(key_data).encode()).hexdigest()[:16]
    
    def _evict_oldest(self):
        """Evict oldest compiled functions."""
        # Sort by call_count (ascending) and evict
        sorted_items = sorted(
            self.compiled.items(),
            key=lambda x: x[1].call_count
        )
        to_remove = len(self.compiled) - self.max_cache_size
        for key, _ in sorted_items[:to_remove]:
            del self.compiled[key]
    
    # -------------------------------------------------------------------------
    # Performance Methods
    # -------------------------------------------------------------------------
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get JIT compilation performance report."""
        total_calls = sum(f.call_count for f in self.compiled.values())
        total_time = sum(f.total_time for f in self.compiled.values())
        
        return {
            'compiled_count': len(self.compiled),
            'compile_count': self.compile_count,
            'total_compile_time': self.total_compile_time,
            'total_calls': total_calls,
            'total_execution_time': total_time,
            'speedup_estimate': (
                self.total_saved_time / self.total_compile_time 
                if self.total_compile_time > 0 else 0
            ),
            'hot_paths': dict(sorted(
                self.hot_paths.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]),
            'numba_available': self.numba_available,
        }
    
    def reset(self):
        """Reset JIT compiler state."""
        self.compiled = {}
        self.hot_paths = {}
        self.total_compile_time = 0.0
        self.total_saved_time = 0.0
        self.compile_count = 0


# =============================================================================
# Auto-Tuner
# =============================================================================

class JITAutoTuner:
    """
    Automatic tuner for JIT compilation parameters.
    
    Adjusts hot threshold and cache size based on runtime metrics.
    """
    
    def __init__(self, compiler: JITCompiler):
        self.compiler = compiler
        self.metrics_history: List[Dict] = []
    
    def tune(self) -> Dict[str, int]:
        """
        Tune JIT parameters based on recent execution.
        
        Returns:
            Dict of tuned parameters
        """
        if len(self.metrics_history) < 10:
            return {'hot_threshold': self.compiler.hot_threshold}
        
        # Analyze recent performance
        avg_calls = np.mean([m.get('calls', 0) for m in self.metrics_history])
        avg_time = np.mean([m.get('time', 0) for m in self.metrics_history])
        
        # Adjust hot threshold
        if avg_time > 1.0:  # Slower execution
            # Lower threshold = more aggressive compilation
            new_threshold = max(10, self.compiler.hot_threshold - 10)
        else:
            # Higher threshold = less compilation overhead
            new_threshold = min(500, self.compiler.hot_threshold + 10)
        
        # Record tuning decision
        self.metrics_history.append({
            'hot_threshold': new_threshold,
            'avg_calls': avg_calls,
            'avg_time': avg_time,
        })
        
        return {'hot_threshold': new_threshold}
    
    def record_execution(self, calls: int, time: float):
        """Record execution metrics for tuning."""
        self.metrics_history.append({'calls': calls, 'time': time})
        
        # Keep only recent history
        if len(self.metrics_history) > 100:
            self.metrics_history = self.metrics_history[-100:]


# =============================================================================
# Convenience Functions
# =============================================================================

def create_jit_compiler(hot_threshold: int = 100) -> JITCompiler:
    """Create a JIT compiler instance."""
    return JITCompiler(hot_threshold=hot_threshold)


# =============================================================================
# Demo/Test
# =============================================================================

if __name__ == '__main__':
    # Create compiler
    compiler = JITCompiler(hot_threshold=10)
    
    print("JIT Compiler Demo")
    print("=" * 40)
    print(f"Numba available: {compiler.numba_available}")
    print(f"Pre-compiled kernels: {len(compiler.compiled)}")
    
    # Mark operations as hot
    for i in range(15):
        compiler.mark_hot('gradient')
    
    print(f"Hot paths: {compiler.hot_paths}")
    
    # Get performance report
    report = compiler.get_performance_report()
    print(f"\nPerformance report:")
    print(f"  Compiled functions: {report['compiled_count']}")
    print(f"  Total compile time: {report['total_compile_time']:.4f}s")
