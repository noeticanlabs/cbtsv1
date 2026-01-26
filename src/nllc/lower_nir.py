from typing import Dict, List, Optional, Tuple
from src.nllc.ast import *
from src.nllc.nir import *
from src.nllc.nir import ObjectType

class Lowerer:
    def __init__(self, file: str):
        self.file = file
        self.value_counter = 0
        self.block_counter = 0
        self.var_env: Dict[str, Value] = {}
        self.module = Module()
        self.current_function: Optional[Function] = None
        self.current_block: Optional[BasicBlock] = None
        self.loop_stack: List[Tuple[str, str]] = []

    def fresh_value(self, ty: Type) -> Value:
        name = f"%{self.value_counter}"
        self.value_counter += 1
        return Value(name, ty)

    def fresh_block(self) -> str:
        name = f"bb{self.block_counter}"
        self.block_counter += 1
        return name

    def add_instruction(self, inst: Instruction):
        assert self.current_block is not None
        self.current_block.instructions.append(inst)

    def is_terminated(self) -> bool:
        if not self.current_block or not self.current_block.instructions:
            return False
        last_inst = self.current_block.instructions[-1]
        return isinstance(last_inst, (BrInst, RetInst))

    def make_trace(self, node: Node, ast_path: str) -> Trace:
        return Trace(self.file, node.span, ast_path)

    def lower_program(self, program: Program) -> Module:
        if len(program.statements) == 1 and program.statements[0].__class__.__name__ == 'FnDecl' and program.statements[0].name == 'main':
            self.lower_fn_decl(program.statements[0], "program.statements[0]")
        else:
            # Wrap top-level statements in a "main" function
            main_func = Function("main", [], IntType(), [])  # assume returns int
            self.module.functions.append(main_func)
            self.current_function = main_func
            entry_block = BasicBlock("entry")
            self.current_function.blocks.append(entry_block)
            self.current_block = entry_block

            for i, stmt in enumerate(program.statements):
                self.lower_statement(stmt, f"program.statements[{i}]")

            # Add return 0
            ret_inst = RetInst(self.make_trace(program, "program"), Value("0", IntType()))
            self.add_instruction(ret_inst)

        return self.module

    def lower_statement(self, stmt: Statement, ast_path: str):
        if stmt.__class__.__name__ == 'FnDecl':
            self.lower_fn_decl(stmt, ast_path)
        elif stmt.__class__.__name__ == 'LetStmt':
            self.lower_let_stmt(stmt, ast_path)
        elif stmt.__class__.__name__ == 'MutStmt':
            self.lower_mut_stmt(stmt, ast_path)
        elif stmt.__class__.__name__ == 'AssignStmt':
            self.lower_assign_stmt(stmt, ast_path)
        elif stmt.__class__.__name__ == 'IfStmt':
            self.lower_if_stmt(stmt, ast_path)
        elif stmt.__class__.__name__ == 'WhileStmt':
            self.lower_while_stmt(stmt, ast_path)
        elif stmt.__class__.__name__ == 'BreakStmt':
            self.lower_break_stmt(stmt, ast_path)
        elif stmt.__class__.__name__ == 'ReturnStmt':
            self.lower_return_stmt(stmt, ast_path)
        elif stmt.__class__.__name__ == 'ExprStmt':
            self.lower_expr_stmt(stmt, ast_path)
        # Ignore others for now

    def lower_fn_decl(self, stmt: FnDecl, ast_path: str):
        # Assume all params int for now
        params = [Value(p, IntType()) for p in stmt.params]
        func = Function(stmt.name, params, IntType(), [])
        self.module.functions.append(func)
        # Lower body later, but for now, set up blocks
        old_function = self.current_function
        old_block = self.current_block
        old_var_env = self.var_env.copy()
        old_value_counter = self.value_counter
        old_block_counter = self.block_counter
        old_loop_stack = self.loop_stack.copy()

        self.current_function = func
        self.var_env = {}
        entry_block = BasicBlock("entry")
        self.current_function.blocks.append(entry_block)
        self.current_block = entry_block
        
        # Allocate stack slots for parameters
        for p in params:
            ptr = self.fresh_value(p.ty)
            alloc_inst = AllocInst(self.make_trace(stmt, ast_path), ptr, p.ty)
            self.add_instruction(alloc_inst)
            store_inst = StoreInst(self.make_trace(stmt, ast_path), ptr, None, p)
            self.add_instruction(store_inst)
            self.var_env[p.name] = ptr

        self.value_counter = 0  # reset for function
        self.block_counter = 0
        self.loop_stack = []

        for i, body_stmt in enumerate(stmt.body):
            self.lower_statement(body_stmt, f"{ast_path}.body[{i}]")

        # If no return, add return none
        if not self.current_block.instructions or not isinstance(self.current_block.instructions[-1], RetInst):
            ret_inst = RetInst(self.make_trace(stmt, ast_path), None)
            self.add_instruction(ret_inst)

        # Restore
        self.current_function = old_function
        self.current_block = old_block
        self.var_env = old_var_env
        self.value_counter = old_value_counter
        self.block_counter = old_block_counter
        self.loop_stack = old_loop_stack

    def lower_let_stmt(self, stmt: LetStmt, ast_path: str):
        val = self.lower_expr(stmt.expr, f"{ast_path}.expr")
        ptr = self.fresh_value(val.ty)
        alloc_inst = AllocInst(self.make_trace(stmt, ast_path), ptr, val.ty)
        self.add_instruction(alloc_inst)
        store_inst = StoreInst(self.make_trace(stmt, ast_path), ptr, None, val)
        self.add_instruction(store_inst)
        self.var_env[stmt.var] = ptr

    def lower_mut_stmt(self, stmt: MutStmt, ast_path: str):
        # Same as let
        self.lower_let_stmt(stmt, ast_path)

    def lower_assign_stmt(self, stmt: AssignStmt, ast_path: str):
        val = self.lower_expr(stmt.expr, f"{ast_path}.expr")
        if stmt.lvalue.__class__.__name__ == 'Var':
            ptr = self.var_env[stmt.lvalue.name]
            store_inst = StoreInst(self.make_trace(stmt, ast_path), ptr, None, val)
            self.add_instruction(store_inst)
        elif stmt.lvalue.__class__.__name__ == 'Index':
            ptr_val = self.lower_expr(stmt.lvalue.array, f"{ast_path}.lvalue.array")
            index_val = self.lower_expr(stmt.lvalue.index, f"{ast_path}.lvalue.index")
            inst = StoreInst(self.make_trace(stmt, ast_path), ptr_val, index_val, val)
            self.add_instruction(inst)
        else:
            raise NotImplementedError("Unsupported lvalue type")

    def lower_if_stmt(self, stmt: IfStmt, ast_path: str):
        cond_val = self.lower_expr(stmt.cond, f"{ast_path}.cond")
        then_block_name = self.fresh_block()
        else_block_name = self.fresh_block() if stmt.else_body else None
        after_block_name = self.fresh_block()

        br_inst = BrInst(self.make_trace(stmt, ast_path), cond_val, then_block_name, else_block_name or after_block_name)
        self.add_instruction(br_inst)

        # Then block
        then_block = BasicBlock(then_block_name)
        self.current_function.blocks.append(then_block)
        old_block = self.current_block
        self.current_block = then_block
        for i, s in enumerate(stmt.body):
            self.lower_statement(s, f"{ast_path}.body[{i}]")
        if not self.is_terminated():
            then_br = BrInst(self.make_trace(stmt, f"{ast_path}.body"), None, after_block_name, None)
            self.add_instruction(then_br)

        # Else block
        if stmt.else_body:
            else_block = BasicBlock(else_block_name)
            self.current_function.blocks.append(else_block)
            self.current_block = else_block
            for i, s in enumerate(stmt.else_body):
                self.lower_statement(s, f"{ast_path}.else_body[{i}]")
            if not self.is_terminated():
                else_br = BrInst(self.make_trace(stmt, f"{ast_path}.else_body"), None, after_block_name, None)
                self.add_instruction(else_br)

        # After block
        after_block = BasicBlock(after_block_name)
        self.current_function.blocks.append(after_block)
        self.current_block = after_block

    def lower_while_stmt(self, stmt: WhileStmt, ast_path: str):
        header_name = self.fresh_block()
        body_name = self.fresh_block()
        after_name = self.fresh_block()

        # Jump to header
        br_to_header = BrInst(self.make_trace(stmt, ast_path), None, header_name, None)
        self.add_instruction(br_to_header)

        # Header
        header_block = BasicBlock(header_name)
        self.current_function.blocks.append(header_block)
        old_block = self.current_block
        self.current_block = header_block
        cond_val = self.lower_expr(stmt.cond, f"{ast_path}.cond")
        br_cond = BrInst(self.make_trace(stmt, f"{ast_path}.cond"), cond_val, body_name, after_name)
        self.add_instruction(br_cond)

        # Body
        body_block = BasicBlock(body_name)
        self.current_function.blocks.append(body_block)
        self.current_block = body_block
        self.loop_stack.append((header_name, after_name))
        for i, s in enumerate(stmt.body):
            self.lower_statement(s, f"{ast_path}.body[{i}]")
        self.loop_stack.pop()
        if not self.is_terminated():
            br_back = BrInst(self.make_trace(stmt, f"{ast_path}.body"), None, header_name, None)
            self.add_instruction(br_back)

        # After
        after_block = BasicBlock(after_name)
        self.current_function.blocks.append(after_block)
        self.current_block = after_block

    def lower_break_stmt(self, stmt: BreakStmt, ast_path: str):
        if not self.loop_stack:
            raise ValueError("Break outside loop")
        _, after = self.loop_stack[-1]
        br_inst = BrInst(self.make_trace(stmt, ast_path), None, after, None)
        self.add_instruction(br_inst)

    def lower_return_stmt(self, stmt: ReturnStmt, ast_path: str):
        val = self.lower_expr(stmt.expr, f"{ast_path}.expr") if stmt.expr else None
        ret_inst = RetInst(self.make_trace(stmt, ast_path), val)
        self.add_instruction(ret_inst)

    def lower_expr_stmt(self, stmt: ExprStmt, ast_path: str):
        if stmt.expr.__class__.__name__ == 'Call':
            args = [self.lower_expr(a, f"{ast_path}.expr.args[{i}]") for i, a in enumerate(stmt.expr.args)]
            if stmt.expr.func in ['ricci_tensor', 'compute_constraints']:
                inst = IntrinsicCallInst(self.make_trace(stmt, ast_path), None, stmt.expr.func, args)
            else:
                inst = CallInst(self.make_trace(stmt, ast_path), None, stmt.expr.func, args)
            self.add_instruction(inst)
        else:
            self.lower_expr(stmt.expr, f"{ast_path}.expr")

    def lower_expr(self, expr: Expr, ast_path: str) -> Value:
        if expr.__class__.__name__ == 'IntLit':
            ty = IntType()
            val = self.fresh_value(ty)
            inst = ConstInst(self.make_trace(expr, ast_path), val, expr.value)
            self.add_instruction(inst)
            return val
        elif expr.__class__.__name__ == 'FloatLit':
            ty = FloatType()
            val = self.fresh_value(ty)
            inst = ConstInst(self.make_trace(expr, ast_path), val, expr.value)
            self.add_instruction(inst)
            return val
        elif expr.__class__.__name__ == 'BoolLit':
            ty = BoolType()
            val = self.fresh_value(ty)
            inst = ConstInst(self.make_trace(expr, ast_path), val, expr.value)
            self.add_instruction(inst)
            return val
        elif expr.__class__.__name__ == 'StrLit':
            ty = StrType()
            val = self.fresh_value(ty)
            inst = ConstInst(self.make_trace(expr, ast_path), val, expr.value)
            self.add_instruction(inst)
            return val
        elif expr.__class__.__name__ == 'ArrayLit':
            # Assume all elements same type
            if expr.elements:
                elem_ty = self.infer_type(expr.elements[0])
                ty = ArrayType(elem_ty)
                val = self.fresh_value(ty)
                values = [self.lower_expr(e, f"{ast_path}.elements[{i}]") for i, e in enumerate(expr.elements)]
                # For simplicity, const with list of values
                inst = ConstInst(self.make_trace(expr, ast_path), val, [v.name for v in values])
                self.add_instruction(inst)
                return val
            else:
                ty = ArrayType(IntType())  # default
                val = self.fresh_value(ty)
                inst = ConstInst(self.make_trace(expr, ast_path), val, [])
                self.add_instruction(inst)
                return val
        elif expr.__class__.__name__ == 'ObjectLit':
            # Assume object as dict
            ty = ObjectType()  # need to define
            val = self.fresh_value(ty)
            fields = {k: self.lower_expr(v, f"{ast_path}.fields[{k}]") for k, v in expr.fields.items()}
            inst = ConstInst(self.make_trace(expr, ast_path), val, {k: v.name for k, v in fields.items()})
            self.add_instruction(inst)
            return val
        elif expr.__class__.__name__ == 'Var':
            ptr = self.var_env[expr.name]
            val = self.fresh_value(ptr.ty) # ptr.ty is the type of the element
            inst = LoadInst(self.make_trace(expr, ast_path), val, ptr)
            self.add_instruction(inst)
            return val
        elif expr.__class__.__name__ == 'Index':
            array_val = self.lower_expr(expr.array, f"{ast_path}.array")
            index_val = self.lower_expr(expr.index, f"{ast_path}.index")
            ty = self.infer_type(expr)  # element type
            val = self.fresh_value(ty)
            inst = GetElementInst(self.make_trace(expr, ast_path), val, array_val, index_val)
            self.add_instruction(inst)
            return val
        elif expr.__class__.__name__ == 'Call':
            args = [self.lower_expr(a, f"{ast_path}.args[{i}]") for i, a in enumerate(expr.args)]
            ty = IntType()  # assume
            result_val = self.fresh_value(ty)
            if expr.func in ['ricci_tensor', 'compute_constraints']:
                inst = IntrinsicCallInst(self.make_trace(expr, ast_path), result_val, expr.func, args)
            else:
                inst = CallInst(self.make_trace(expr, ast_path), result_val, expr.func, args)
            self.add_instruction(inst)
            return result_val
        elif expr.__class__.__name__ == 'BinOp':
            left_val = self.lower_expr(expr.left, f"{ast_path}.left")
            right_val = self.lower_expr(expr.right, f"{ast_path}.right")
            ty = self.infer_type(expr)
            val = self.fresh_value(ty)
            inst = BinOpInst(self.make_trace(expr, ast_path), val, left_val, expr.op, right_val)
            self.add_instruction(inst)
            return val
        elif expr.__class__.__name__ == 'IfExpr':
            ty = self.infer_type(expr)
            temp_val = self.fresh_value(ty)
            alloc_inst = AllocInst(self.make_trace(expr, ast_path), temp_val, ty)
            self.add_instruction(alloc_inst)
            then_block_name = self.fresh_block()
            else_block_name = self.fresh_block() if expr.else_body else None
            after_block_name = self.fresh_block()
            cond_val = self.lower_expr(expr.cond, f"{ast_path}.cond")
            br_inst = BrInst(self.make_trace(expr, ast_path), cond_val, then_block_name, else_block_name or after_block_name)
            self.add_instruction(br_inst)
            # Then
            then_block = BasicBlock(then_block_name)
            self.current_function.blocks.append(then_block)
            old_current = self.current_block
            self.current_block = then_block
            then_val = self.lower_expr(expr.body, f"{ast_path}.body")
            store_then = StoreInst(self.make_trace(expr, f"{ast_path}.then_body"), temp_val, None, then_val)
            self.add_instruction(store_then)
            br_then = BrInst(self.make_trace(expr, f"{ast_path}.then_body"), None, after_block_name, None)
            self.add_instruction(br_then)
            # Else
            if expr.else_body:
                else_block = BasicBlock(else_block_name)
                self.current_function.blocks.append(else_block)
                self.current_block = else_block
                else_val = self.lower_expr(expr.else_body, f"{ast_path}.else_body")
                store_else = StoreInst(self.make_trace(expr, f"{ast_path}.else_body"), temp_val, None, else_val)
                self.add_instruction(store_else)
                br_else = BrInst(self.make_trace(expr, f"{ast_path}.else_body"), None, after_block_name, None)
                self.add_instruction(br_else)
            # After
            after_block = BasicBlock(after_block_name)
            self.current_function.blocks.append(after_block)
            self.current_block = after_block
            # Load
            loaded_val = self.fresh_value(ty)
            load_inst = LoadInst(self.make_trace(expr, ast_path), loaded_val, temp_val)
            self.add_instruction(load_inst)
            return loaded_val
        else:
            raise NotImplementedError(f"Unsupported expr {type(expr)}")

    def infer_type(self, expr: Expr) -> Type:
        if isinstance(expr, (IntLit, BinOp)):
            return IntType()
        elif isinstance(expr, FloatLit):
            return FloatType()
        elif isinstance(expr, BoolLit):
            return BoolType()
        elif isinstance(expr, StrLit):
            return StrType()
        elif isinstance(expr, ArrayLit):
            if expr.elements:
                return ArrayType(self.infer_type(expr.elements[0]))
            return ArrayType(IntType())
        elif isinstance(expr, ObjectLit):
            return ObjectType()
        elif isinstance(expr, Index):
            if isinstance(expr.array, ArrayLit):
                return self.infer_type(expr.array.elements[0]) if expr.array.elements else IntType()
            elif isinstance(expr.array, ObjectLit):
                # Assume str key, any value, but for simplicity, any
                return IntType()
            # Assume known
            return IntType()
        elif isinstance(expr, IfExpr):
            return self.infer_type(expr.body)
        elif isinstance(expr, Call):
            return IntType()  # assume
        elif isinstance(expr, Var):
            return self.var_env[expr.name].ty
        else:
            return IntType()