"""
NSC-M3L ALG Model: Symbolic Simplifier

Implements expression simplification to canonical form.
"""

from typing import List, Dict, Set, Optional, Tuple
from dataclasses import dataclass

from .alg_types import (
    AlgebraicExpr, RingElement, FieldElement, Matrix, Vector,
    Polynomial, Monomial, RationalFunction,
    Sum, Product, Power, Commutator, Bracket,
    is_scalar, is_matrix, is_polynomial
)


@dataclass
class SimplificationRule:
    """Pattern-matching rule for simplification."""
    pattern: str  # Pattern string for matching
    replacement: 'AlgebraicExpr'  # Replacement expression
    condition: Optional[str] = None  # Condition for applicability


class SymbolicSimplifier:
    """Simplify algebraic expressions to canonical form."""
    
    def __init__(self):
        self.rules: List[SimplificationRule] = []
        self._init_default_rules()
    
    def _init_default_rules(self):
        """Initialize default simplification rules."""
        # Zero rules
        self.rules.append(SimplificationRule(
            pattern="x + 0",
            replacement=Sum(terms=[])  # Identity
        ))
        self.rules.append(SimplificationRule(
            pattern="0 + x",
            replacement=Sum(terms=[])  # Identity
        ))
        self.rules.append(SimplificationRule(
            pattern="x * 0",
            replacement=RingElement(0)
        ))
        self.rules.append(SimplificationRule(
            pattern="0 * x",
            replacement=RingElement(0)
        ))
        
        # One rules
        self.rules.append(SimplificationRule(
            pattern="x * 1",
            replacement=Sum(terms=[])  # Identity
        ))
        self.rules.append(SimplificationRule(
            pattern="1 * x",
            replacement=Sum(terms=[])  # Identity
        ))
        self.rules.append(SimplificationRule(
            pattern="x^1",
            replacement=Sum(terms=[])  # Identity
        ))
        
        # Power rules
        self.rules.append(SimplificationRule(
            pattern="x^0",
            replacement=RingElement(1)
        ))
        # x^(-1) rule removed - requires RationalFunction with proper variables
    
    def simplify(self, expr: AlgebraicExpr) -> AlgebraicExpr:
        """Simplify expression to normal form."""
        if is_scalar(expr):
            return self._simplify_scalar(expr)
        elif is_matrix(expr):
            return self._simplify_matrix(expr)
        elif is_polynomial(expr):
            return self._simplify_polynomial(expr)
        elif isinstance(expr, Sum):
            return self._simplify_sum(expr)
        elif isinstance(expr, Product):
            return self._simplify_product(expr)
        elif isinstance(expr, Power):
            return self._simplify_power(expr)
        elif isinstance(expr, Commutator):
            return self._simplify_commutator(expr)
        else:
            return expr
    
    def _simplify_scalar(self, expr: RingElement) -> AlgebraicExpr:
        """Simplify scalar (no-op for now)."""
        return expr
    
    def _simplify_matrix(self, expr: Matrix) -> Matrix:
        """Simplify matrix elements."""
        new_data = []
        for row in expr.data:
            new_row = []
            for elem in row:
                simplified = self.simplify(elem)
                if isinstance(simplified, RingElement):
                    new_row.append(simplified)
                else:
                    new_row.append(elem)
            new_data.append(new_row)
        return Matrix(data=new_data, ring=expr.ring)
    
    def _simplify_polynomial(self, expr: Polynomial) -> Polynomial:
        """Simplify polynomial by combining like terms."""
        # This delegates to the Polynomial.simplify() method
        return expr.simplify()
    
    def _simplify_sum(self, expr: Sum) -> AlgebraicExpr:
        """Simplify sum by combining like terms."""
        simplified_terms = []
        for term in expr.terms:
            s = self.simplify(term)
            if isinstance(s, RingElement) and s.value == 0:
                continue
            if isinstance(s, Sum):
                # Flatten nested sums
                simplified_terms.extend(s.terms)
            else:
                simplified_terms.append(s)
        
        if not simplified_terms:
            return RingElement(0)
        if len(simplified_terms) == 1:
            return simplified_terms[0]
        
        return Sum(terms=simplified_terms)
    
    def _simplify_product(self, expr: Product) -> AlgebraicExpr:
        """Simplify product by combining constants and eliminating ones/zeros."""
        simplified_factors = []
        has_zero = False
        constant_product = RingElement(1)
        
        for factor in expr.factors:
            s = self.simplify(factor)
            
            if isinstance(s, RingElement):
                if s.value == 0:
                    has_zero = True
                    break
                constant_product = RingElement(constant_product.value * s.value)
                continue
            
            if isinstance(s, Product):
                # Flatten nested products
                simplified_factors.extend(s.factors)
            else:
                simplified_factors.append(s)
        
        if has_zero:
            return RingElement(0)
        
        # Combine constant with first factor if exists
        if constant_product.value != 1 and simplified_factors:
            simplified_factors[0] = Product(factors=[
                RingElement(constant_product.value),
                simplified_factors[0]
            ])
        elif constant_product.value != 1:
            return RingElement(constant_product.value)
        
        if not simplified_factors:
            return RingElement(1)
        if len(simplified_factors) == 1:
            return simplified_factors[0]
        
        return Product(factors=simplified_factors)
    
    def _simplify_power(self, expr: Power) -> AlgebraicExpr:
        """Simplify power expression."""
        base = self.simplify(expr.base)
        
        # x^0 = 1
        if expr.exponent == 0:
            return RingElement(1)
        
        # x^1 = x
        if expr.exponent == 1:
            return base
        
        # (x^n)^m = x^(n*m)
        if isinstance(base, Power):
            return Power(base=base.base, exponent=base.exponent * expr.exponent)
        
        return Power(base=base, exponent=expr.exponent)
    
    def _simplify_commutator(self, expr: Commutator) -> AlgebraicExpr:
        """Simplify commutator."""
        a = self.simplify(expr.a)
        b = self.simplify(expr.b)
        
        # [X, X] = 0
        if a == b:
            return RingElement(0)
        
        return Commutator(a=a, b=b)
    
    def expand(self, expr: AlgebraicExpr) -> AlgebraicExpr:
        """Expand products and powers."""
        if isinstance(expr, Product):
            return self._expand_product(expr)
        elif isinstance(expr, Power):
            return self._expand_power(expr)
        elif isinstance(expr, Sum):
            return Sum(terms=[self.expand(t) for t in expr.terms])
        else:
            return expr
    
    def _expand_product(self, expr: Product) -> AlgebraicExpr:
        """Expand product into sum."""
        if len(expr.factors) != 2:
            # For more than 2 factors, expand pairwise
            first, *rest = expr.factors
            if rest:
                expanded_rest = self.expand(Product(factors=rest))
                return self._expand_product(Product(factors=[first, expanded_rest]))
            return expr
        
        a, b = expr.factors
        
        # (A + B)(C + D) = AC + AD + BC + BD
        if isinstance(a, Sum) and isinstance(b, Sum):
            terms = []
            for t1 in a.terms:
                for t2 in b.terms:
                    terms.append(Product(factors=[t1, t2]))
            return self.expand(Sum(terms=terms))
        
        # A(B + C) = AB + AC
        if isinstance(b, Sum):
            terms = [Product(factors=[a, t]) for t in b.terms]
            return self.expand(Sum(terms=terms))
        
        # (A + B)C = AC + BC
        if isinstance(a, Sum):
            terms = [Product(factors=[t, b]) for t in a.terms]
            return self.expand(Sum(terms=terms))
        
        # (AB)^n expansion using binomial theorem would go here
        return expr
    
    def _expand_power(self, expr: Power) -> AlgebraicExpr:
        """Expand power expression."""
        base = expr.base
        exp = expr.exponent
        
        if isinstance(base, Sum) and exp > 0:
            # Use binomial expansion for (A+B)^n
            return self._binomial_expand(base, exp)
        
        return Power(base=base, exponent=exp)
    
    def _binomial_expand(self, base: Sum, n: int) -> AlgebraicExpr:
        """Binomial expansion of (A+B)^n."""
        if n == 0:
            return RingElement(1)
        if n == 1:
            return base
        
        # Simple case: (a+b)^2 = a^2 + 2ab + b^2
        if n == 2 and len(base.terms) == 2:
            a, b = base.terms
            a2 = self.expand(Power(base=a, exponent=2))
            b2 = self.expand(Power(base=b, exponent=2))
            ab = self.expand(Product(factors=[a, b]))
            two_ab = Product(factors=[RingElement(2), ab])
            return Sum(terms=[a2, two_ab, b2])
        
        # Fall back to direct computation
        return Power(base=base, exponent=n)
    
    def factor(self, expr: AlgebraicExpr) -> AlgebraicExpr:
        """Factor expressions."""
        if not is_polynomial(expr):
            return expr
        
        polynomial = expr
        # Simple factoring: look for common factors
        return self._factor_polynomial(polynomial)
    
    def _factor_polynomial(self, poly: Polynomial) -> Polynomial:
        """Factor polynomial (placeholder implementation)."""
        # This is a placeholder - full polynomial factoring requires
        # more sophisticated algorithms (Berlekamp, etc.)
        return poly
    
    def collect_terms(self, expr: AlgebraicExpr, vars: List[str]) -> AlgebraicExpr:
        """Collect like terms with respect to specified variables."""
        if not is_polynomial(expr):
            return expr
        
        polynomial = expr
        # Group terms by monomials in specified variables
        collected: Dict[Tuple[int, ...], List[Monomial]] = {}
        
        for term in polynomial.terms:
            key = []
            for var in vars:
                if var in term.variables:
                    idx = term.variables.index(var)
                    key.append(term.exponents[idx])
                else:
                    key.append(0)
            key = tuple(key)
            if key not in collected:
                collected[key] = []
            collected[key].append(term)
        
        # Combine terms in each group
        new_terms = []
        for key, terms in collected.items():
            if len(terms) == 1:
                new_terms.append(terms[0])
            else:
                # Sum of terms with same variables
                combined_coeff = RingElement(0)
                for term in terms:
                    combined_coeff = RingElement(combined_coeff.value + term.coefficient.value)
                new_terms.append(Monomial(
                    variables=vars,
                    exponents=list(key),
                    coefficient=combined_coeff
                ))
        
        return Polynomial(terms=new_terms)
    
    def canonicalize(self, expr: AlgebraicExpr) -> AlgebraicExpr:
        """Put expression in canonical form."""
        # Simplify first
        simplified = self.simplify(expr)
        
        # Then expand
        expanded = self.expand(simplified)
        
        # Then simplify again
        return self.simplify(expanded)
    
    def normalize_fraction(self, expr: AlgebraicExpr) -> AlgebraicExpr:
        """Normalize rational function."""
        if not isinstance(expr, RationalFunction):
            return expr
        
        num = self.simplify(expr.numerator)
        den = self.simplify(expr.denominator)
        
        # Compute GCD and divide (placeholder)
        return RationalFunction(numerator=num, denominator=den)
    
    def equivalence(self, expr1: AlgebraicExpr, expr2: AlgebraicExpr) -> bool:
        """Check algebraic equivalence."""
        # Simplify both and compare
        s1 = self.canonicalize(expr1)
        s2 = self.canonicalize(expr2)
        
        return s1 == s2


def simplify(expr: AlgebraicExpr) -> AlgebraicExpr:
    """Convenience function to simplify an expression."""
    simplifier = SymbolicSimplifier()
    return simplifier.simplify(expr)


def expand(expr: AlgebraicExpr) -> AlgebraicExpr:
    """Convenience function to expand an expression."""
    simplifier = SymbolicSimplifier()
    return simplifier.expand(expr)


def canonicalize(expr: AlgebraicExpr) -> AlgebraicExpr:
    """Convenience function to canonicalize an expression."""
    simplifier = SymbolicSimplifier()
    return simplifier.canonicalize(expr)
