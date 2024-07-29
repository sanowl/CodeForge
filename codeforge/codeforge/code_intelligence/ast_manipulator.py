import ast, astor, logging
from typing import List, Union, Callable

logger = logging.getLogger(__name__)

class ASTManipulator:
    def __init__(self): self.transformations = []

    def modify(self, code: str) -> str:
        tree = ast.parse(code)
        for transform in self.transformations: tree = transform(tree)
        return astor.to_source(tree)

    def add_function(self, func_def: str) -> 'ASTManipulator':
        self.transformations.append(lambda tree: self._add_node(tree, ast.parse(func_def).body[0]))
        return self

    def remove_function(self, func_name: str) -> 'ASTManipulator':
        self.transformations.append(lambda tree: self._remove_node(tree, lambda n: isinstance(n, ast.FunctionDef) and n.name == func_name))
        return self

    def rename_function(self, old_name: str, new_name: str) -> 'ASTManipulator':
        def rename(node):
            if isinstance(node, ast.FunctionDef) and node.name == old_name: node.name = new_name
            elif isinstance(node, ast.Name) and node.id == old_name: node.id = new_name
            return node
        self.transformations.append(lambda tree: self._transform_nodes(tree, rename))
        return self

    def add_import(self, import_stmt: str) -> 'ASTManipulator':
        self.transformations.append(lambda tree: self._add_node(tree, ast.parse(import_stmt).body[0], 0))
        return self

    def remove_import(self, module_name: str) -> 'ASTManipulator':
        self.transformations.append(lambda tree: self._remove_node(tree, lambda n: isinstance(n, (ast.Import, ast.ImportFrom)) and any(alias.name == module_name for alias in n.names)))
        return self

    def add_decorator(self, func_name: str, decorator: str) -> 'ASTManipulator':
        def add_dec(node):
            if isinstance(node, ast.FunctionDef) and node.name == func_name:
                node.decorator_list.append(ast.Name(id=decorator, ctx=ast.Load()))
            return node
        self.transformations.append(lambda tree: self._transform_nodes(tree, add_dec))
        return self

    def inline_function(self, func_name: str) -> 'ASTManipulator':
        def inline(tree):
            func_def = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == func_name), None)
            if func_def:
                calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == func_name]
                for call in calls:
                    call_args = {arg.arg: arg.value for arg in call.keywords}
                    call_args.update({func_def.args.args[i].arg: call.args[i] for i in range(len(call.args))})
                    inlined = [ast.fix_missing_locations(ast.copy_deepcopy(stmt)) for stmt in func_def.body]
                    for node in ast.walk(ast.Module(inlined)):
                        if isinstance(node, ast.Name) and node.id in call_args:
                            node.id = call_args[node.id].id if isinstance(call_args[node.id], ast.Name) else node.id
                    call.parent.body[call.parent.body.index(call)] = inlined[-1]
                    call.parent.body[call.parent.body.index(call):call.parent.body.index(call)] = inlined[:-1]
                tree.body = [n for n in tree.body if n != func_def]
            return tree
        self.transformations.append(inline)
        return self

    @staticmethod
    def _add_node(tree: ast.AST, node: ast.AST, index: int = -1) -> ast.AST:
        tree.body.insert(index if index >= 0 else len(tree.body), node)
        return tree

    @staticmethod
    def _remove_node(tree: ast.AST, condition: Callable[[ast.AST], bool]) -> ast.AST:
        tree.body = [node for node in tree.body if not condition(node)]
        return tree

    @staticmethod
    def _transform_nodes(tree: ast.AST, transform: Callable[[ast.AST], ast.AST]) -> ast.AST:
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                transformed = transform(child)
                for field, old_value in ast.iter_fields(node):
                    if isinstance(old_value, list):
                        for i, value in enumerate(old_value):
                            if value == child:
                                old_value[i] = transformed
                    elif old_value == child:
                        setattr(node, field, transformed)
        return tree