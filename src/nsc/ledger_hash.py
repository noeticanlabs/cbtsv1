"""Hash chain implementation for receipt integrity verification.

This module provides the HashChain class for maintaining receipt integrity
through cryptographic hash chaining, ensuring that all receipts in a ledger
are tamper-evident.
"""

import hashlib
from typing import Dict, List, Optional, Tuple, Any


class HashChain:
    """Hash chain for maintaining receipt integrity.
    
    The hash chain creates a linked structure of receipts where each receipt's
    hash includes the hash of the previous receipt, creating a tamper-evident
    chain. This allows detection of any modification to receipt history.
    
    Attributes:
        algorithm: The hash algorithm to use (default: sha256)
        chain: List of hashes in the chain
        genesis_hash: The genesis (first) hash for the chain
    """
    
    VALID_ALGORITHMS = {"sha256", "sha512", "sha3_256", "sha3_512"}
    
    def __init__(self, algorithm: str = "sha256"):
        """Initialize the hash chain.
        
        Args:
            algorithm: Hash algorithm to use. Must be one of the valid algorithms.
            
        Raises:
            ValueError: If an invalid algorithm is specified.
        """
        if algorithm not in self.VALID_ALGORITHMS:
            raise ValueError(f"Invalid hash algorithm: {algorithm}. "
                           f"Must be one of {self.VALID_ALGORITHMS}")
        self.algorithm = algorithm
        self.chain: List[str] = []
        self.genesis_hash = self._compute_genesis()
    
    def _compute_genesis(self) -> str:
        """Compute the genesis hash for the chain.
        
        The genesis hash serves as the anchor point for the hash chain.
        It is computed from a constant string to ensure reproducibility.
        
        Returns:
            The computed genesis hash string.
        """
        genesis_data = f"GENESIS:{self.algorithm}:NSC-M3L-LEDGER"
        return self._compute_hash_bytes(genesis_data.encode())
    
    def _compute_hash_bytes(self, data: bytes) -> str:
        """Compute hash using the configured algorithm.
        
        Args:
            data: The bytes data to hash.
            
        Returns:
            The hexadecimal hash string.
        """
        if self.algorithm == "sha256":
            return hashlib.sha256(data).hexdigest()
        elif self.algorithm == "sha512":
            return hashlib.sha512(data).hexdigest()
        elif self.algorithm == "sha3_256":
            return hashlib.sha3_256(data).hexdigest()
        elif self.algorithm == "sha3_512":
            return hashlib.sha3_512(data).hexdigest()
        else:
            # Fallback to sha256
            return hashlib.sha256(data).hexdigest()
    
    def compute_hash(self, data: Dict[str, Any], 
                     prev_hash: Optional[str] = None) -> str:
        """Compute hash for receipt data.
        
        Creates a deterministic hash from receipt data, including the previous
        hash in the chain to create linkage.
        
        Args:
            data: Dictionary of receipt data to hash.
            prev_hash: The previous hash in the chain (optional).
            
        Returns:
            The computed hash string.
        """
        # Create a deterministic serialization of the data
        # Sort keys for reproducibility
        sorted_items = self._serialize_for_hash(data)
        
        # Convert to string representation
        data_str = str(sorted_items)
        
        # Include previous hash if provided
        if prev_hash:
            data_str = f"{data_str}:{prev_hash}"
        
        # Include algorithm identifier for domain separation
        data_str = f"{self.algorithm}:{data_str}"
        
        return self._compute_hash_bytes(data_str.encode())
    
    def _serialize_for_hash(self, data: Any) -> Any:
        """Serialize data for hashing, handling complex types.
        
        Recursively processes data to create a hashable representation,
        handling lists, dicts, and other types appropriately.
        
        Args:
            data: The data to serialize.
            
        Returns:
            A hashable representation of the data.
        """
        if isinstance(data, dict):
            return tuple(sorted((k, self._serialize_for_hash(v)) 
                               for k, v in data.items()))
        elif isinstance(data, list):
            return tuple(self._serialize_for_hash(item) for item in data)
        elif isinstance(data, set):
            return tuple(sorted(self._serialize_for_hash(item) 
                               for item in data))
        elif isinstance(data, bytes):
            return data.hex()
        elif hasattr(data, 'tolist'):
            # numpy arrays
            return tuple(data.tolist())
        else:
            return data
    
    def serialize_for_hash(self, receipt: Any) -> Dict[str, Any]:
        """Serialize a receipt object for hashing.
        
        Extracts the relevant fields from a receipt object, excluding
        the hash fields themselves.
        
        Args:
            receipt: The receipt object to serialize.
            
        Returns:
            Dictionary of receipt data suitable for hashing.
        """
        if hasattr(receipt, '__dict__'):
            # Extract from object
            return {
                k: v for k, v in receipt.__dict__.items()
                if k not in ('hash', 'hash_prev')
            }
        elif isinstance(receipt, dict):
            # Already a dict
            return {
                k: v for k, v in receipt.items()
                if k not in ('hash', 'hash_prev')
            }
        else:
            raise TypeError(f"Cannot serialize type: {type(receipt)}")
    
    def validate_chain(self, receipts: List[Any]) -> Tuple[bool, List[str]]:
        """Validate hash chain continuity.
        
        Verifies that all receipts in the chain are properly linked
        and that hashes are valid.
        
        Args:
            receipts: List of receipts to validate.
            
        Returns:
            Tuple of (is_valid, list_of_errors).
        """
        errors = []
        prev_hash = self.genesis_hash
        
        for i, rcpt in enumerate(receipts):
            # Check that hash_prev matches the expected previous hash
            if rcpt.hash_prev != prev_hash:
                errors.append(f"Chain break at receipt {i}: "
                            f"expected prev_hash={prev_hash[:16]}..., "
                            f"got {rcpt.hash_prev[:16] if rcpt.hash_prev else 'None'}...")
            
            # Serialize and compute expected hash
            try:
                data = self.serialize_for_hash(rcpt)
                computed = self.compute_hash(data, rcpt.hash_prev)
            except Exception as e:
                errors.append(f"Error computing hash at receipt {i}: {e}")
                computed = None
            
            # Check hash matches
            if computed is not None and rcpt.hash != computed:
                errors.append(f"Hash mismatch at receipt {i}: "
                            f"expected={computed[:16]}..., "
                            f"got={rcpt.hash[:16] if rcpt.hash else 'None'}...")
            
            # Update prev_hash for next iteration
            if rcpt.hash:
                prev_hash = rcpt.hash
        
        return len(errors) == 0, errors
    
    def validate_partial_chain(self, 
                                receipts: List[Any],
                                expected_start_hash: str) -> Tuple[bool, List[str]]:
        """Validate a partial hash chain.
        
        Similar to validate_chain but allows starting from an arbitrary point
        in the chain rather than the genesis.
        
        Args:
            receipts: List of receipts to validate.
            expected_start_hash: The expected hash preceding the first receipt.
            
        Returns:
            Tuple of (is_valid, list_of_errors).
        """
        errors = []
        prev_hash = expected_start_hash
        
        for i, rcpt in enumerate(receipts):
            if rcpt.hash_prev != prev_hash:
                errors.append(f"Chain break at receipt {i}")
            
            try:
                data = self.serialize_for_hash(rcpt)
                computed = self.compute_hash(data, rcpt.hash_prev)
            except Exception as e:
                errors.append(f"Error computing hash at receipt {i}: {e}")
                computed = None
            
            if computed is not None and rcpt.hash != computed:
                errors.append(f"Hash mismatch at receipt {i}")
            
            if rcpt.hash:
                prev_hash = rcpt.hash
        
        return len(errors) == 0, errors
    
    def get_chain_length(self) -> int:
        """Get the current length of the chain."""
        return len(self.chain)
    
    def get_last_hash(self) -> str:
        """Get the last hash in the chain."""
        if self.chain:
            return self.chain[-1]
        return self.genesis_hash
    
    def reset(self) -> None:
        """Reset the hash chain to genesis state."""
        self.chain = []
    
    def compute_receipt_hash(self, receipt: Any) -> str:
        """Compute and set the hash for a receipt.
        
        Takes a receipt object (with hash_prev already set) and computes
        its hash, setting the hash attribute.
        
        Args:
            receipt: The receipt to hash.
            
        Returns:
            The computed hash.
        """
        data = self.serialize_for_hash(receipt)
        receipt.hash = self.compute_hash(data, receipt.hash_prev)
        return receipt.hash
