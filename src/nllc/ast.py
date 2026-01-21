from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class Span:
    start: int
    end: int

@dataclass
class Node:
    span: Span

@dataclass
class Program(Node):
    statements: List['Statement']

@dataclass
class Statement(Node):
    pass

@dataclass
class ImportStmt(Statement):
    name: str

@dataclass
class LetStmt(Statement):
    var: str
    expr: 'Expr'

@dataclass
class MutStmt(Statement):
    var: str
    expr: 'Expr'

@dataclass
class AssignStmt(Statement):
    lvalue: 'Expr'
    expr: 'Expr'

@dataclass
class IfStmt(Statement):
    cond: 'Expr'
    body: List[Statement]
    else_body: Optional[List[Statement]] = None

@dataclass
class WhileStmt(Statement):
    cond: 'Expr'
    body: List[Statement]

@dataclass
class ReturnStmt(Statement):
    expr: Optional['Expr'] = None

@dataclass
class BreakStmt(Statement):
    pass

@dataclass
class ExprStmt(Statement):
    expr: 'Expr'

@dataclass
class FnDecl(Statement):
    name: str
    params: List[str]
    body: List[Statement]

@dataclass
class ThreadBlock(Statement):
    domain: str
    scale: str
    phase: str
    body: List[Statement]
    require: List[str]

@dataclass
class Expr(Node):
    pass

@dataclass
class IntLit(Expr):
    value: int

@dataclass
class FloatLit(Expr):
    value: float

@dataclass
class BoolLit(Expr):
    value: bool

@dataclass
class StrLit(Expr):
    value: str

@dataclass
class ArrayLit(Expr):
    elements: List[Expr]

@dataclass
class ObjectLit(Expr):
    fields: dict

@dataclass
class Var(Expr):
    name: str

@dataclass
class Index(Expr):
    array: Expr
    index: Expr

@dataclass
class Call(Expr):
    func: str
    args: List[Expr]

@dataclass
class BinOp(Expr):
    left: Expr
    op: str
    right: Expr

@dataclass
class IfExpr(Expr):
    cond: Expr
    body: Expr
    else_body: Optional[Expr] = None