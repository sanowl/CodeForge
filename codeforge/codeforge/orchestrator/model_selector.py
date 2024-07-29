import logging
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from codeforge.config import HUGGINGFACE_API_TOKEN
import torch
from typing import Dict, Any
import ast
import re
import numpy as np
from collections import defaultdict

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
        self.models = {
            "code_generation": ["gpt4", "claude", "codex", "copilot"],
            "code_analysis": ["codebert", "gpt4", "claude", "codex"],
            "refactoring": ["gpt4", "claude", "codex", "copilot"],
            "bug_fixing": ["copilot", "gpt4", "claude", "codex"]
        }
        self.model_performance = defaultdict(lambda: defaultdict(lambda: 0.5))

    async def select_model(self, task: Dict[str, Any]) -> str:
        logger.info(f"Selecting model for task: {task['task_type']}")
        
        task_type = await self._classify_task(task['content'])
        sentiment = await self._analyze_sentiment(task['content'])
        complexity = await self._estimate_complexity(task['content'])
        code_quality = await self._assess_code_quality(task['content'])
        
        suitable_models = self.models.get(task_type, self.models["code_generation"])
        scores = await self._score_models(suitable_models, task_type, sentiment, complexity, code_quality)
        
        selected_model = max(scores, key=scores.get)
        logger.info(f"Selected model: {selected_model} for task type: {task_type}")
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

    async def _estimate_complexity(self, content: str) -> float:
        ast_complexity = self._calculate_ast_complexity(content)
        lexical_complexity = self._calculate_lexical_complexity(content)
        structural_complexity = self._calculate_structural_complexity(content)
        
        total_complexity = (
            0.4 * ast_complexity +
            0.3 * lexical_complexity +
            0.3 * structural_complexity
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

    async def _assess_code_quality(self, content: str) -> float:
        inputs = self.code_tokenizer(content, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.code_model(**inputs)
        scores = torch.nn.functional.softmax(outputs.logits, dim=1)
        return scores[0][1].item()  # Assuming binary classification (good/bad code)

    async def _score_models(self, models: list, task_type: str, sentiment: float, complexity: float, code_quality: float) -> Dict[str, float]:
        base_scores = {"gpt4": 0.9, "claude": 0.85, "codex": 0.8, "codebert": 0.75, "copilot": 0.7}
        scores = {}
        for model in models:
            performance = self.model_performance[task_type][model]
            score = (
                base_scores[model] * 0.3 +
                sentiment * 0.1 +
                (1 - complexity) * 0.2 +  # Lower complexity is better
                code_quality * 0.2 +
                performance * 0.2
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