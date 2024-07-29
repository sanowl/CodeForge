import logging
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from codeforge.config import HUGGINGFACE_API_TOKEN
import torch
from typing import Dict, Any, Tuple
import ast
import re
import numpy as np
from collections import defaultdict
import pygments
from pygments.lexers import get_lexer_by_name, guess_lexer
from pygments.util import ClassNotFound

logger = logging.getLogger(__name__)

class ComplexityVisitor(ast.NodeVisitor):
    def __init__(self):
        self.complexity = 1
        self.depth = 0

    def visit(self, node):
        self.depth += 1
        if isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor, ast.ExceptHandler)):
            self.complexity += self.depth
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            self.complexity += self.depth * 2
        super().visit(node)
        self.depth -= 1

class ModelSelector:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english", token=HUGGINGFACE_API_TOKEN)
        self.sentiment_model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english", token=HUGGINGFACE_API_TOKEN)
        self.task_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", token=HUGGINGFACE_API_TOKEN)
        self.code_tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base", token=HUGGINGFACE_API_TOKEN)
        self.code_model = AutoModelForSequenceClassification.from_pretrained("microsoft/codebert-base", token=HUGGINGFACE_API_TOKEN)
        self.language_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", token=HUGGINGFACE_API_TOKEN)
        self.framework_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", token=HUGGINGFACE_API_TOKEN)
        
        self.models = {
            "code_generation": ["gpt4", "claude", "codex", "copilot"],
            "code_analysis": ["codebert", "gpt4", "claude", "codex"],
            "refactoring": ["gpt4", "claude", "codex", "copilot"],
            "bug_fixing": ["copilot", "gpt4", "claude", "codex"]
        }
        self.model_performance = defaultdict(lambda: defaultdict(lambda: 0.5))
        
        self.language_specific_models = {
            "python": {"analysis": "huggingface/CodeBERTa-small-v1", "generation": "codeparrot/codeparrot-small"},
            "javascript": {"analysis": "microsoft/codebert-base", "generation": "openai-codex"},
            "java": {"analysis": "microsoft/codebert-base-mlm", "generation": "openai-codex"},
            "cpp": {"analysis": "microsoft/codebert-base", "generation": "openai-codex"},
            "ruby": {"analysis": "huggingface/CodeBERTa-small-v1", "generation": "openai-codex"},
        }
        
        self.framework_specific_models = {
            "react": "huggingface/CodeBERTa-small-v1",
            "django": "codeparrot/codeparrot-small",
            "spring": "microsoft/codebert-base-mlm",
            "angular": "microsoft/codebert-base",
            "flask": "huggingface/CodeBERTa-small-v1",
        }

    async def select_model(self, task: Dict[str, Any]) -> str:
        logger.info(f"Selecting model for task: {task['task_type']}")
        
        task_type = await self._classify_task(task['content'])
        sentiment = await self._analyze_sentiment(task['content'])
        language, framework = await self._detect_language_and_framework(task['content'])
        complexity = await self._estimate_complexity(task['content'], language)
        code_quality = await self._assess_code_quality(task['content'], language)
        
        suitable_models = self.models.get(task_type, self.models["code_generation"])
        scores = await self._score_models(suitable_models, task_type, sentiment, complexity, code_quality, language, framework)
        
        selected_model = max(scores, key=scores.get)
        logger.info(f"Selected model: {selected_model} for task type: {task_type}, language: {language}, framework: {framework}")
        return selected_model

    async def _classify_task(self, content: str) -> str:
        result = await self.task_classifier(content, list(self.models.keys()))
        return result['labels'][0]

    async def _analyze_sentiment(self, content: str) -> float:
        inputs = self.tokenizer(content, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.sentiment_model(**inputs)
        scores = torch.nn.functional.softmax(outputs.logits, dim=1)
        return scores[0][1].item()  # Positive sentiment score

    async def _detect_language_and_framework(self, content: str) -> Tuple[str, str]:
        languages = ["python", "javascript", "java", "cpp", "ruby"]
        frameworks = ["react", "django", "spring", "angular", "flask"]
        
        try:
            lexer = guess_lexer(content)
            detected_language = lexer.name.lower()
        except ClassNotFound:
            language_result = await self.language_classifier(content, languages)
            detected_language = language_result['labels'][0]
        
        framework_result = await self.framework_classifier(content, frameworks)
        detected_framework = framework_result['labels'][0]
        
        return detected_language, detected_framework

    async def _estimate_complexity(self, content: str, language: str) -> float:
        if language == "python":
            return self._calculate_python_complexity(content)
        else:
            return self._calculate_generic_complexity(content)

    def _calculate_python_complexity(self, content: str) -> float:
        ast_complexity = self._calculate_ast_complexity(content)
        lexical_complexity = self._calculate_lexical_complexity(content)
        structural_complexity = self._calculate_structural_complexity(content)
        
        total_complexity = (
            0.4 * ast_complexity +
            0.3 * lexical_complexity +
            0.3 * structural_complexity
        )
        
        return min(1.0, total_complexity / 100)  # Normalize to [0, 1]

    def _calculate_generic_complexity(self, content: str) -> float:
        lexical_complexity = self._calculate_lexical_complexity(content)
        structural_complexity = self._calculate_structural_complexity(content)
        
        total_complexity = (
            0.5 * lexical_complexity +
            0.5 * structural_complexity
        )
        
        return min(1.0, total_complexity / 100)  # Normalize to [0, 1]

    def _calculate_ast_complexity(self, content: str) -> float:
        try:
            tree = ast.parse(content)
            visitor = ComplexityVisitor()
            visitor.visit(tree)
            return visitor.complexity
        except SyntaxError:
            return 50  # Default high complexity for invalid syntax

    def _calculate_lexical_complexity(self, content: str) -> float:
        words = re.findall(r'\w+', content)
        avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
        unique_words = len(set(words))
        return (avg_word_length * 5) + (unique_words * 0.1)

    def _calculate_structural_complexity(self, content: str) -> float:
        lines = content.split('\n')
        indentation_levels = [len(line) - len(line.lstrip()) for line in lines if line.strip()]
        max_indentation = max(indentation_levels) if indentation_levels else 0
        avg_indentation = sum(indentation_levels) / len(indentation_levels) if indentation_levels else 0
        return (max_indentation * 2) + (avg_indentation * 3)

    async def _assess_code_quality(self, content: str, language: str) -> float:
        model_name = self.language_specific_models.get(language, {}).get("analysis", "microsoft/codebert-base")
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=HUGGINGFACE_API_TOKEN)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, token=HUGGINGFACE_API_TOKEN)
        
        inputs = tokenizer(content, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        scores = torch.nn.functional.softmax(outputs.logits, dim=1)
        return scores[0][1].item()  # Assuming binary classification (good/bad code)

    async def _score_models(self, models: list, task_type: str, sentiment: float, complexity: float, code_quality: float, language: str, framework: str) -> Dict[str, float]:
        base_scores = {"gpt4": 0.9, "claude": 0.85, "codex": 0.8, "codebert": 0.75, "copilot": 0.7}
        scores = {}
        for model in models:
            performance = self.model_performance[task_type][model]
            language_boost = 0.1 if model in self.language_specific_models.get(language, {}).values() else 0
            framework_boost = 0.1 if model == self.framework_specific_models.get(framework, "") else 0
            score = (
                base_scores[model] * 0.3 +
                sentiment * 0.1 +
                (1 - complexity) * 0.2 +  # Lower complexity is better
                code_quality * 0.2 +
                performance * 0.1 +
                language_boost +
                framework_boost
            )
            scores[model] = score
        return scores

    async def update_model_performance(self, model: str, task_type: str, performance_score: float):
        current_performance = self.model_performance[task_type][model]
        new_performance = current_performance * 0.7 + performance_score * 0.3  # Exponential moving average
        self.model_performance[task_type][model] = new_performance

        if task_type in self.models and model in self.models[task_type]:
            sorted_models = sorted(self.models[task_type], key=lambda m: self.model_performance[task_type][m], reverse=True)
            self.models[task_type] = sorted_models

        logger.info(f"Updated model rankings for {task_type}: {self.models[task_type]}")