import logging
from transformers import pipeline
from codeforge.config import HUGGINGFACE_API_TOKEN

logger = logging.getLogger(__name__)

class EntityExtractor:
    def __init__(self):
        self.extractor = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english", token=HUGGINGFACE_API_TOKEN)

    async def extract(self, content: str):
        logger.info(f"Extracting entities from content: {content[:50]}...")
        
        entities = self.extractor(content)
        
        # Group entities by type
        grouped_entities = {}
        for entity in entities:
            if entity['entity'] not in grouped_entities:
                grouped_entities[entity['entity']] = []
            grouped_entities[entity['entity']].append(entity['word'])
        
        logger.info(f"Extracted entities: {grouped_entities}")
        
        return grouped_entities