"""
NSC-M3L ALG Model: Gröbner Basis

Implements Gröbner basis computation and polynomial system solving.
"""

from typing import List, Optional, Set, Dict, Tuple
from dataclasses import dataclass
from collections import defaultdict

from .alg_types import (
    Polynomial, Monomial, RingElement, AlgebraicExpr
)


@dataclass
class TermOrder:
    """Term ordering for Gröbner basis computation."""
    # Supported orderings: 'lex', 'grlex', 'grevlex'
    ordering: str = 'lex'
    variables: List[str] = None  # Variable ordering for lex
    
    def __post_init__(self):
        if self.variables is None:
            self.variables = []
    
    def compare(self, m1: Monomial, m2: Monomial) -> int:
        """Compare two monomials. Returns -1 if m1 < m2, 1 if m1 > m2, 0 if equal."""
        if self.ordering == 'lex':
            return self._lex_compare(m1, m2)
        elif self.ordering == 'grlex':
            return self._grlex_compare(m1, m2)
        elif self.ordering == 'grevlex':
            return self._grevlex_compare(m1, m2)
        return 0
    
    def _lex_compare(self, m1: Monomial, m2: Monomial) -> int:
        """Lexicographic comparison."""
        for var in self.variables:
            if var not in m1.variables and var not in m2.variables:
                continue
            e1 = m1.exponents[m1.variables.index(var)] if var in m1.variables else 0
            e2 = m2.exponents[m2.variables.index(var)] if var in m2.variables else 0
            if e1 < e2:
                return -1
            if e1 > e2:
                return 1
        return 0
    
    def _grlex_compare(self, m1: Monomial, m2: Monomial) -> int:
        """Graded lex comparison (total degree first, then lex)."""
        d1, d2 = m1.degree(), m2.degree()
        if d1 < d2:
            return -1
        if d1 > d2:
            return 1
        return self._lex_compare(m1, m2)
    
    def _grevlex_compare(self, m1: Monomial, m2: Monomial) -> int:
        """Graded reverse lex comparison."""
        d1, d2 = m1.degree(), m2.degree()
        if d1 < d2:
            return -1
        if d1 > d2:
            return 1
        # Reverse lex: compare last variable first
        vars_ = m1.variables
        for i in range(len(vars_) - 1, -1, -1):
            e1 = m1.exponents[i] if i < len(m1.exponents) else 0
            e2 = m2.exponents[i] if i < len(m2.exponents) else 0
            if e1 < e2:
                return 1  # Reverse: smaller exponent means larger
            if e1 > e2:
                return -1
        return 0


class GroebnerBasis:
    """Gröbner basis computation."""
    
    def __init__(self, term_order: Optional[TermOrder] = None):
        self.order = term_order or TermOrder(ordering='lex')
    
    def compute_basis(self, polynomials: List[Polynomial]) -> List[Polynomial]:
        """
        Compute Gröbner basis using Buchberger's algorithm.
        
        Args:
            polynomials: List of polynomials generating the ideal
            
        Returns:
            Gröbner basis of the ideal
        """
        if not polynomials:
            return []
        
        # Collect all variables
        all_vars = set()
        for poly in polynomials:
            for term in poly.terms:
                all_vars.update(term.variables)
        
        self.order.variables = list(all_vars)
        
        # Initialize basis with input polynomials
        G = [p for p in polynomials if self._nonzero(p)]
        
        # Buchberger's algorithm
        changed = True
        while changed:
            changed = False
            pairs = self._generate_pairs(G)
            
            for (f, g) in pairs:
                S = self._S_polynomial(f, g)
                remainder = self.reduce(S, G)
                
                if self._nonzero(remainder):
                    G.append(remainder)
                    changed = True
        
        # Reduce basis to minimal Gröbner basis
        G = self._minimalize(G)
        
        return G
    
    def _generate_pairs(self, polys: List[Polynomial]) -> List[Tuple[Polynomial, Polynomial]]:
        """Generate all pairs of polynomials."""
        pairs = []
        n = len(polys)
        for i in range(n):
            for j in range(i + 1, n):
                pairs.append((polys[i], polys[j]))
        return pairs
    
    def _S_polynomial(self, f: Polynomial, g: Polynomial) -> Polynomial:
        """Compute S-polynomial of f and g."""
        LT_f = self._leading_term(f)
        LT_g = self._leading_term(g)
        
        # Find LCM of leading terms
        lcm = self._lcm_monomial(LT_f, LT_g)
        
        # S(f,g) = (lcm/LT_f)*f - (lcm/LT_g)*g
        coeff_f = RingElement(lcm.coefficient.value / LT_f.coefficient.value)
        coeff_g = RingElement(lcm.coefficient.value / LT_g.coefficient.value)
        
        term_f = Monomial(
            variables=lcm.variables,
            exponents=[e1 - e2 for e1, e2 in zip(lcm.exponents, LT_f.exponents)],
            coefficient=coeff_f
        )
        term_g = Monomial(
            variables=lcm.variables,
            exponents=[e1 - e2 for e1, e2 in zip(lcm.exponents, LT_g.exponents)],
            coefficient=coeff_g
        )
        
        f_scaled = self._multiply_monomial(f, term_f)
        g_scaled = self._multiply_monomial(g, term_g)
        
        return self._subtract(f_scaled, g_scaled)
    
    def _multiply_monomial(self, poly: Polynomial, mono: Monomial) -> Polynomial:
        """Multiply polynomial by monomial."""
        new_terms = []
        for term in poly.terms:
            # Multiply exponents
            new_vars = list(set(term.variables + mono.variables))
            new_exponents = []
            for var in new_vars:
                e1 = term.exponents[term.variables.index(var)] if var in term.variables else 0
                e2 = mono.exponents[mono.variables.index(var)] if var in mono.variables else 0
                new_exponents.append(e1 + e2)
            
            new_coeff = RingElement(term.coefficient.value * mono.coefficient.value)
            new_terms.append(Monomial(
                variables=new_vars,
                exponents=new_exponents,
                coefficient=new_coeff
            ))
        
        return Polynomial(terms=new_terms)
    
    def _subtract(self, f: Polynomial, g: Polynomial) -> Polynomial:
        """Subtract polynomials."""
        terms = list(f.terms)
        
        for g_term in g.terms:
            found = False
            for i, f_term in enumerate(terms):
                if (f_term.variables == g_term.variables and 
                    f_term.exponents == g_term.exponents):
                    new_coeff = RingElement(f_term.coefficient.value - g_term.coefficient.value)
                    if new_coeff.value != 0:
                        terms[i] = Monomial(
                            variables=f_term.variables,
                            exponents=f_term.exponents,
                            coefficient=new_coeff
                        )
                    else:
                        terms.pop(i)
                    found = True
                    break
            
            if not found:
                terms.append(Monomial(
                    variables=g_term.variables,
                    exponents=g_term.exponents,
                    coefficient=RingElement(-g_term.coefficient.value)
                ))
        
        return Polynomial(terms=terms)
    
    def _lcm_monomial(self, m1: Monomial, m2: Monomial) -> Monomial:
        """Compute LCM of two monomials."""
        all_vars = list(set(m1.variables + m2.variables))
        all_vars.sort()  # Deterministic ordering
        
        new_exponents = []
        for var in all_vars:
            e1 = m1.exponents[m1.variables.index(var)] if var in m1.variables else 0
            e2 = m2.exponents[m2.variables.index(var)] if var in m2.variables else 0
            new_exponents.append(max(e1, e2))
        
        return Monomial(
            variables=all_vars,
            exponents=new_exponents,
            coefficient=RingElement(1)
        )
    
    def reduce(self, poly: Polynomial, basis: List[Polynomial]) -> Polynomial:
        """
        Reduce polynomial modulo Gröbner basis.
        
        Returns remainder with respect to the basis.
        """
        remainder = Polynomial(terms=list(poly.terms))
        changed = True
        
        while changed and self._nonzero(remainder):
            changed = False
            
            for g in basis:
                LT_g = self._leading_term(g)
                
                while True:
                    LT_r = self._leading_term(remainder)
                    
                    # Check if LT(remainder) is divisible by LT(g)
                    divisible, factor = self._divides(LT_g, LT_r)
                    
                    if divisible:
                        # remainder = remainder - factor * g
                        factor_mono = Monomial(
                            variables=LT_r.variables,
                            exponents=[e1 - e2 for e1, e2 in zip(LT_r.exponents, LT_g.exponents)],
                            coefficient=RingElement(factor)
                        )
                        scaled_g = self._multiply_monomial(g, factor_mono)
                        remainder = self._subtract(remainder, scaled_g)
                        changed = True
                        
                        if self._nonzero(remainder):
                            break
                    else:
                        break
            
            if not changed:
                break
        
        return remainder
    
    def _divides(self, m1: Monomial, m2: Monomial) -> Tuple[bool, float]:
        """Check if m1 divides m2."""
        for var in m1.variables:
            if var not in m2.variables:
                return False, 0
            e1 = m1.exponents[m1.variables.index(var)]
            e2 = m2.exponents[m2.variables.index(var)]
            if e1 > e2:
                return False, 0
        
        # Check coefficient
        if m2.coefficient.value == 0:
            return False, 0
        factor = m2.coefficient.value / m1.coefficient.value
        
        return True, factor
    
    def _leading_term(self, poly: Polynomial) -> Monomial:
        """Get leading term with respect to term order."""
        if not poly.terms:
            return Monomial(variables=[], exponents=[], coefficient=RingElement(0))
        
        leading = poly.terms[0]
        for term in poly.terms[1:]:
            if self.order.compare(term, leading) > 0:
                leading = term
        return leading
    
    def _nonzero(self, poly: Polynomial) -> bool:
        """Check if polynomial is nonzero."""
        return len(poly.terms) > 0 and all(
            t.coefficient.value != 0 for t in poly.terms
        )
    
    def _minimalize(self, basis: List[Polynomial]) -> List[Polynomial]:
        """Remove redundant polynomials from basis."""
        minimal = []
        
        for i, g in enumerate(basis):
            # Check if g is redundant with respect to others
            others = minimal.copy()
            remainder = self.reduce(g, others)
            
            if self._nonzero(remainder):
                # Divide out leading coefficients
                LT_g = self._leading_term(g)
                if LT_g.coefficient.value != 1:
                    factor = LT_g.coefficient.value
                    new_terms = []
                    for term in g.terms:
                        new_terms.append(Monomial(
                            variables=term.variables,
                            exponents=term.exponents,
                            coefficient=RingElement(term.coefficient.value / factor)
                        ))
                    g = Polynomial(terms=new_terms)
                minimal.append(g)
        
        return minimal
    
    def solve_ideal(
        self, 
        basis: List[Polynomial], 
        variables: List[str]
    ) -> List[Dict[str, float]]:
        """
        Solve polynomial system using Gröbner basis.
        
        Returns approximate solutions.
        """
        if not basis:
            return []
        
        # Build elimination ideal
        eliminated = self._eliminate_variables(basis, variables)
        
        # For now, return empty solutions
        # Full solving would require numerical methods or algebraic techniques
        return []
    
    def _eliminate_variables(
        self, 
        basis: List[Polynomial],
        vars_to_eliminate: List[str]
    ) -> List[Polynomial]:
        """Compute elimination ideal."""
        # This would compute the part of the Gröbner basis
        # that doesn't contain certain variables
        return basis
    
    def elimination_ideal(
        self, 
        basis: List[Polynomial], 
        vars_to_eliminate: List[str]
    ) -> List[Polynomial]:
        """
        Compute elimination ideal for given variables.
        
        Uses F4/F4-style elimination.
        """
        # Build basis with new variable ordering
        remaining_vars = [v for v in self.order.variables if v not in vars_to_eliminate]
        self.order.variables = vars_to_eliminate + remaining_vars
        
        # Recompute Gröbner basis
        G = self.compute_basis(basis)
        
        # Return polynomials with only remaining variables
        eliminated = []
        for poly in G:
            poly_vars = set()
            for term in poly.terms:
                poly_vars.update(term.variables)
            
            if not poly_vars.intersection(set(vars_to_eliminate)):
                eliminated.append(poly)
        
        return eliminated


def compute_groebner_basis(
    polynomials: List[Polynomial],
    ordering: str = 'lex'
) -> List[Polynomial]:
    """Convenience function to compute Gröbner basis."""
    order = TermOrder(ordering=ordering)
    gb = GroebnerBasis(term_order=order)
    return gb.compute_basis(polynomials)
