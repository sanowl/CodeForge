import logging, ast, symtable
from typing import Dict, Any
from pylint import lint
from pylint.reporters import JSONReporter
from io import StringIO

logger = logging.getLogger(__name__)

class BugDetector:
    def __init__(self): self.ast_analyzer, self.symtable_analyzer, self.pylint_analyzer = ASTAnalyzer(), SymtableAnalyzer(), PylintAnalyzer()

    async def detect_bugs(self, code: str) -> Dict[str, Any]:
        logger.info("Detecting bugs in code")
        bugs, warnings, metrics = [], [], {}
        for analyzer in [self.ast_analyzer, self.symtable_analyzer, self.pylint_analyzer]:
            result = await analyzer.analyze(code)
            bugs.extend(result.get('bugs', [])); warnings.extend(result.get('warnings', [])); metrics.update(result.get('metrics', {}))
        logger.info(f"Detected {len(bugs)} bugs and {len(warnings)} warnings")
        return {'bugs': bugs, 'warnings': warnings, 'metrics': metrics}

class ASTAnalyzer:
    async def analyze(self, code: str) -> Dict[str, Any]:
        bugs, warnings, metrics = [], [], {'num_functions': 0, 'num_classes': 0, 'num_imports': 0, 'complexity': 0}
        try: tree = ast.parse(code)
        except SyntaxError as e: return {'bugs': [f"Syntax error: {str(e)}"], 'warnings': [], 'metrics': metrics}
        for node in ast.walk(tree):
            if isinstance(node, ast.Try) and not node.handlers: bugs.append(f"Empty except block at line {node.lineno}")
            elif isinstance(node, ast.Except) and isinstance(node.type, ast.Name) and node.type.id == 'Exception': warnings.append(f"Broad exception handler at line {node.lineno}")
            elif isinstance(node, ast.FunctionDef): metrics['num_functions'] += 1; metrics['complexity'] += self.calculate_complexity(node)
            elif isinstance(node, ast.ClassDef): metrics['num_classes'] += 1
            elif isinstance(node, (ast.Import, ast.ImportFrom)): metrics['num_imports'] += 1
        return {'bugs': bugs, 'warnings': warnings, 'metrics': metrics}

    def calculate_complexity(self, node: ast.AST) -> int:
        return 1 + sum(self.calculate_complexity(child) if isinstance(child, ast.FunctionDef) else 1
                       for child in ast.iter_child_nodes(node)
                       if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor, ast.ExceptHandler, ast.With, ast.AsyncWith, ast.FunctionDef)))

class SymtableAnalyzer:
    async def analyze(self, code: str) -> Dict[str, Any]:
        warnings = []
        try:
            table = symtable.symtable(code, 'string', 'exec')
            self.check_unused_variables(table, warnings)
        except SyntaxError: pass
        return {'warnings': warnings}

    def check_unused_variables(self, table: symtable.SymbolTable, warnings: list):
        for name in table.get_identifiers():
            symbol = table.lookup(name)
            if symbol.is_referenced() is False and symbol.is_parameter() is False and symbol.is_imported() is False:
                warnings.append(f"Unused variable '{name}' at line {symbol.get_lineno()}")
        for child_table in table.get_children(): self.check_unused_variables(child_table, warnings)

class PylintAnalyzer:
    async def analyze(self, code: str) -> Dict[str, Any]:
        pylint_output = StringIO()
        lint.Run(['--output-format=json', '--from-stdin'], stdin=code, reporter=JSONReporter(pylint_output), exit=False)
        pylint_result = eval(pylint_output.getvalue())
        bugs = [f"{item['message']} at line {item['line']}" for item in pylint_result if item['type'] in ('error', 'fatal')]
        warnings = [f"{item['message']} at line {item['line']}" for item in pylint_result if item['type'] in ('warning', 'convention', 'refactor')]
        metrics = {'pylint_score': 10 - sum(item['score'] for item in pylint_result)}
        return {'bugs': bugs, 'warnings': warnings, 'metrics': metrics}