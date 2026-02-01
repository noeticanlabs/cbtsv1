"""
NSC-M3L ALG Model: Expression Rewriting

Implements pattern-based expression rewriting system.
"""

from typing import List, Dict, Set, Optional, Tuple, Callable
from dataclasses import dataclass, field
import re

from .alg_types import (
    AlgebraicExpr, RingElement, Matrix, Vector,
    Polynomial, Monomial, Sum, Product, Power, Commutator, Bracket,
    is_scalar, is_matrix, is_polynomial
)


@dataclass
class RewriteRule:
    """A single rewrite rule."""
    pattern: str  # Pattern string
    replacement: 'AlgebraicExpr'  # Replacement expression
    condition: Optional[Callable[[AlgebraicExpr], bool]] = None
    priority: int = 0  # Higher priority rules applied first


class ExprRewriter:
    """Expression rewriting with pattern matching."""
    
    def __init__(self):
        self.rules: List[RewriteRule] = []
        self._init_builtin_rules()
    
    def _init_builtin_rules(self):
        """Initialize built-in rewrite rules."""
        # Arithmetic rules
        self.add_rule("0 + x → x")
        self.add_rule("x + 0 → x")
        self.add_rule("x - x → 0")
        self.add_rule("0 * x → 0")
        self.add_rule("x * 0 → 0")
        self.add_rule("1 * x → x")
        self.add_rule("x * 1 → x")
        self.add_rule("x / 1 → x")
        self.add_rule("0 / x → 0")
        
        # Power rules
        self.add_rule("x^0 → 1")
        self.add_rule("x^1 → x")
        self.add_rule("1^x → 1")
        self.add_rule("x^(-1) → 1/x")
        self.add_rule("(x^2) → x*x")
        
        # Commutator rules
        self.add_rule("[x, x] → 0")
        self.add_rule("[x, y] + [y, x] → 0")
        self.add_rule("[x, y + z] → [x, y] + [x, z]")
        self.add_rule("[x + y, z] → [x, z] + [y, z]")
        self.add_rule("[x, [y, z]] + [y, [z, x]] + [z, [x, y]] → 0", priority=10)  # Jacobi
        
        # Matrix rules
        self.add_rule("I * x → x")  # Identity
        self.add_rule("x * I → x")
        self.add_rule("det(I) → 1")
        self.add_rule("trace(I) → n")
    
    def add_rule(self, rule_str: str, priority: int = 0):
        """
        Add a rule in pattern → replacement format.
        
        Examples:
            "x + 0 → x"
            "x^2 → x * x"
            "[x, x] → 0"
        """
        if "→" in rule_str:
            pattern_str, replacement_str = rule_str.split("→")
            pattern_str = pattern_str.strip()
            replacement_str = replacement_str.strip()
        else:
            pattern_str = rule_str
            replacement_str = ""
        
        self.rules.append(RewriteRule(
            pattern=pattern_str,
            replacement=self._parse_replacement(replacement_str),
            priority=priority
        ))
    
    def _parse_replacement(self, s: str) -> AlgebraicExpr:
        """Parse replacement string to expression."""
        s = s.strip()
        
        if not s or s == "0":
            return RingElement(0)
        if s == "1":
            return RingElement(1)
        
        # Try to parse as number
        try:
            if "." in s:
                return RingElement(float(s))
            return RingElement(int(s))
        except ValueError:
            pass
        
        # Placeholder for complex parsing
        return RingElement(0)
    
    def add_custom_rule(
        self,
        pattern: str,
        replacement: AlgebraicExpr,
        condition: Optional[Callable[[AlgebraicExpr], bool]] = None,
        priority: int = 0
    ):
        """Add a custom rule with explicit pattern and replacement."""
        self.rules.append(RewriteRule(
            pattern=pattern,
            replacement=replacement,
            condition=condition,
            priority=priority
        ))
    
    def rewrite(self, expr: AlgebraicExpr) -> AlgebraicExpr:
        """Apply all rules to expression until fixed point."""
        # Sort rules by priority (highest first)
        sorted_rules = sorted(self.rules, key=lambda r: -r.priority)
        
        result = expr
        changed = True
        max_iterations = 100
        
        iteration = 0
        while changed and iteration < max_iterations:
            changed = False
            iteration += 1
            
            new_result = self._apply_rules(result, sorted_rules)
            
            if new_result != result:
                changed = True
                result = new_result
        
        return result
    
    def _apply_rules(
        self, 
        expr: AlgebraicExpr, 
        rules: List[RewriteRule]
    ) -> AlgebraicExpr:
        """Apply rules to expression recursively."""
        # Handle different expression types
        if isinstance(expr, Sum):
            new_terms = [self._apply_rules(t, rules) for t in expr.terms]
            return Sum(terms=new_terms)
        
        elif isinstance(expr, Product):
            new_factors = [self._apply_rules(f, rules) for f in expr.factors]
            return Product(factors=new_factors)
        
        elif isinstance(expr, Power):
            new_base = self._apply_rules(expr.base, rules)
            return Power(base=new_base, exponent=expr.exponent)
        
        elif isinstance(expr, Commutator):
            new_a = self._apply_rules(expr.a, rules)
            new_b = self._apply_rules(expr.b, rules)
            return Commutator(a=new_a, b=new_b)
        
        elif isinstance(expr, Bracket):
            new_a = self._apply_rules(expr.a, rules)
            new_b = self._apply_rules(expr.b, rules)
            return Bracket(a=new_a, b=new_b, bracket_type=expr.bracket_type)
        
        # Apply pattern matching to leaf expressions
        for rule in rules:
            if self._matches(expr, rule.pattern):
                if rule.condition is None or rule.condition(expr):
                    return rule.replacement
        
        return expr
    
    def _matches(self, expr: AlgebraicExpr, pattern: str) -> bool:
        """Check if expression matches pattern."""
        # Handle simple patterns
        pattern = pattern.strip()
        
        # Numeric patterns
        if pattern == "0":
            if isinstance(expr, RingElement):
                return expr.value == 0
        if pattern == "1":
            if isinstance(expr, RingElement):
                return expr.value == 1
        
        # Variable pattern (starts with lowercase)
        if pattern.islower() or pattern.startswith("_"):
            return True  # Wildcard match
        
        # Commutator pattern
        if pattern.startswith("[") and pattern.endswith("]"):
            return isinstance(expr, Commutator)
        
        return str(expr) == pattern
    
    def normalize(self, expr: AlgebraicExpr) -> AlgebraicExpr:
        """Normalize expression to canonical form."""
        # Step 1: Expand
        expanded = self._expand(expr)
        
        # Step 2: Apply rewriting
        rewritten = self.rewrite(expanded)
        
        # Step 3: Collect like terms
        collected = self._collect_terms(rewritten)
        
        return collected
    
    def _expand(self, expr: AlgebraicExpr) -> AlgebraicExpr:
        """Expand nested expressions."""
        if isinstance(expr, Sum):
            expanded_terms = []
            for term in expr.terms:
                expanded = self._expand(term)
                if isinstance(expanded, Sum):
                    expanded_terms.extend(expanded.terms)
                else:
                    expanded_terms.append(expanded)
            return Sum(terms=expanded_terms)
        
        elif isinstance(expr, Product):
            # Distribute over sums
            factors = expr.factors
            for i, f in enumerate(factors):
                expanded_f = self._expand(f)
                if isinstance(expanded_f, Sum):
                    # (A+B)*rest = A*rest + B*rest
                    new_factors = list(factors)
                    new_factors[i] = expanded_f.terms[0]
                    result = Product(factors=new_factors)
                    for j in range(1, len(expanded_f.terms)):
                        new_factors = list(factors)
                        new_factors[i] = expanded_f.terms[j]
                        result = Sum(terms=[result, Product(factors=new_factors)])
                    return result
            return expr
        
        else:
            return expr
    
    def _collect_terms(self, expr: AlgebraicExpr) -> AlgebraicExpr:
        """Collect like terms in sums."""
        if not isinstance(expr, Sum):
            return expr
        
        # Group terms
        term_groups: Dict[str, List[AlgebraicExpr]] = {}
        
        for term in expr.terms:
            key = str(term)
            if key not in term_groups:
                term_groups[key] = []
            term_groups[key].append(term)
        
        # Combine terms
        new_terms = []
        for key, terms in term_groups.items():
            if len(terms) == 1:
                new_terms.append(terms[0])
            elif len(terms) == 2:
                # Check if they can be combined
                if isinstance(terms[0], RingElement) and isinstance(terms[1], RingElement):
                    combined = RingElement(terms[0].value + terms[1].value)
                    if combined.value != 0:
                        new_terms.append(combined)
                else:
                    new_terms.extend(terms)
            else:
                new_terms.extend(terms)
        
        if not new_terms:
            return RingElement(0)
        if len(new_terms) == 1:
            return new_terms[0]
        
        return Sum(terms=new_terms)
    
    def equivalence(self, expr1: AlgebraicExpr, expr2: AlgebraicExpr) -> bool:
        """Check algebraic equivalence by normalizing both."""
        norm1 = self.normalize(expr1)
        norm2 = self.normalize(expr2)
        return norm1 == norm2
    
    def substitute(
        self, 
        expr: AlgebraicExpr, 
        substitutions: Dict[str, AlgebraicExpr]
    ) -> AlgebraicExpr:
        """Substitute variables in expression."""
        if isinstance(expr, RingElement):
            return expr
        
        elif isinstance(expr, Sum):
            new_terms = [self.substitute(t, substitutions) for t in expr.terms]
            return Sum(terms=new_terms)
        
        elif isinstance(expr, Product):
            new_factors = [self.substitute(f, substitutions) for f in expr.factors]
            return Product(factors=new_factors)
        
        elif isinstance(expr, Power):
            new_base = self.substitute(expr.base, substitutions)
            return Power(base=new_base, exponent=expr.exponent)
        
        elif isinstance(expr, Commutator):
            new_a = self.substitute(expr.a, substitutions)
            new_b = self.substitute(expr.b, substitutions)
            return Commutator(a=new_a, b=new_b)
        
        return expr


# Convenience function
def rewrite(expr: AlgebraicExpr) -> AlgebraicExpr:
    """Convenience function to rewrite an expression."""
    rewriter = ExprRewriter()
    return rewriter.rewrite(expr)


def normalize(expr: AlgebraicExpr) -> AlgebraicExpr:
    """Convenience function to normalize an expression."""
    rewriter = ExprRewriter()
    return rewriter.normalize(expr)
