"""
Intrinsic Binder for NIR

Binds NIR intrinsic calls to JIT-compiled kernels for GPU/HPC acceleration.
Integrates with the JIT kernel registry from jit_kernel.py.

Usage:
    from src.nllc.intrinsic_binder import IntrinsicBinder
    
    binder = IntrinsicBinder()
    if binder.is_bound('inv_sym6'):
        result = binder.get_kernel('inv_sym6')(arg1, arg2)
"""

from typing import Callable, Dict, Any, Optional
from src.nllc.jit_kernel import KERNEL_REGISTRY, get_kernel


class IntrinsicBinder:
    """
    Binds NIR intrinsic calls to compiled Numba/NumPy kernels.
    
    This class manages the mapping from intrinsic function names to
    their JIT-compiled implementations. It integrates with the kernel
    registry to provide transparent kernel binding.
    
    Usage:
        binder = IntrinsicBinder()
        
        # Check if an intrinsic is bound
        if binder.is_bound('inv_sym6'):
            kernel = binder.get_kernel('inv_sym6')
            result = kernel(arg1, arg2)
        
        # Bind a custom kernel
        binder.bind_intrinsic('custom_op', my_custom_kernel)
    """
    
    def __init__(self):
        self.bound_intrinsics: Dict[str, Callable] = {}
        self._initialize_default_kernels()
    
    def _initialize_default_kernels(self):
        """Initialize default kernels from the registry."""
        # Import all kernels from jit_kernel.py
        # The KERNEL_REGISTRY is populated when jit_kernel.py is imported
        for name, kernel in KERNEL_REGISTRY.items():
            self.bound_intrinsics[name] = kernel
    
    def bind_intrinsic(self, name: str, kernel_func: Callable) -> None:
        """
        Bind an intrinsic name to a kernel function.
        
        Args:
            name: Name of the intrinsic (e.g., 'inv_sym6')
            kernel_func: Callable that implements the intrinsic
        """
        self.bound_intrinsics[name] = kernel_func
    
    def is_bound(self, name: str) -> bool:
        """
        Check if an intrinsic is bound to a kernel.
        
        Args:
            name: Name of the intrinsic
            
        Returns:
            True if bound, False otherwise
        """
        return name in self.bound_intrinsics
    
    def get_kernel(self, name: str) -> Optional[Callable]:
        """
        Get the kernel function for an intrinsic.
        
        Args:
            name: Name of the intrinsic
            
        Returns:
            Kernel function, or None if not bound
        """
        return self.bound_intrinsics.get(name)
    
    def call_intrinsic(self, name: str, args: list) -> Any:
        """
        Call an intrinsic by name with the given arguments.
        
        Args:
            name: Name of the intrinsic
            args: Arguments to pass to the kernel
            
        Returns:
            Result from the kernel execution
            
        Raises:
            ValueError: If the intrinsic is not bound
        """
        kernel = self.get_kernel(name)
        if kernel is None:
            raise ValueError(f"Intrinsic '{name}' is not bound to a kernel")
        return kernel(*args)
    
    def list_intrinsics(self) -> list:
        """
        List all bound intrinsics.
        
        Returns:
            List of intrinsic names
        """
        return list(self.bound_intrinsics.keys())


# Convenience function for VM integration
def get_intrinsic_binder() -> IntrinsicBinder:
    """Get a singleton IntrinsicBinder instance."""
    return IntrinsicBinder()
