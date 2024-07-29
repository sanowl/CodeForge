import logging
from codeforge.nlu.intent_classifier import IntentClassifier
from codeforge.nlu.entity_extractor import EntityExtractor

logger = logging.getLogger(__name__)

class TaskAnalyzer:
    def __init__(self):
        self.intent_classifier = IntentClassifier()
        self.entity_extractor = EntityExtractor()

    async def analyze(self, task_type: str, content: str):
        logger.info(f"Analyzing task: {task_type}")
        
        intent = await self.intent_classifier.classify(content)
        entities = await self.entity_extractor.extract(content)
        
        return {
            "task_type": task_type,
            "intent": intent,
            "entities": entities
        }