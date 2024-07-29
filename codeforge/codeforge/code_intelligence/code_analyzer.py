import logging, ast, builtins
from typing import Dict, Any
from collections import defaultdict

logger = logging.getLogger(__name__)

class CodeAnalyzer:
    async def analyze(self, code: str) -> Dict[str, Any]:
        logger.info("Analyzing code")
        try: tree = ast.parse(code)
        except SyntaxError as e: return {"error": f"Syntax error: {str(e)}"}
        
        analysis = defaultdict(int)
        complexity = 0
        imports, classes, functions = set(), set(), set()
        
        for node in ast.walk(tree):
            node_type = type(node).__name__
            analysis[f"num_{node_type.lower()}s"] += 1
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                complexity += self._calculate_complexity(node)
                functions.add(node.name)
            elif isinstance(node, ast.ClassDef): classes.add(node.name)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                imports.update(alias.name for alias in node.names)
        
        analysis.update({
            "complexity": complexity,
            "imports": list(imports),
            "classes": list(classes),
            "functions": list(functions),
            "builtin_usage": self._count_builtin_usage(tree),
            "depth": self._calculate_max_depth(tree),
            "lines_of_code": len(code.splitlines())
        })
        logger.info(f"Analysis result: {dict(analysis)}")
        return dict(analysis)

    def _calculate_complexity(self, node: ast.AST) -> int:
        return 1 + sum(self._calculate_complexity(child) for child in ast.iter_child_nodes(node)
                       if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor, ast.ExceptHandler, ast.With, ast.AsyncWith)))

    def _count_builtin_usage(self, tree: ast.AST) -> Dict[str, int]:
        return {name: len([n for n in ast.walk(tree) if isinstance(n, ast.Name) and n.id == name])
                for name in dir(builtins) if not name.startswith('_')}

    def _calculate_max_depth(self, node: ast.AST, current_depth: int = 0) -> int:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            return current_depth
        return max(self._calculate_max_depth(child, current_depth + 1) for child in ast.iter_child_nodes(node))