import logging
import openai
from codeforge.config import OPENAI_API_KEY

openai.api_key = OPENAI_API_KEY

logger = logging.getLogger(__name__)

class QueryExpander:
    async def expand(self, query: str):
        logger.info(f"Expanding query: {query}")
        
        response = await openai.Completion.create(
            engine="text-davinci-002",
            prompt=f"Expand the following query with more details:\n{query}\n\nExpanded query:",
            max_tokens=100,
            n=1,
            stop=None,
            temperature=0.5,
        )
        
        expanded_query = response.choices[0].text.strip()
        
        logger.info(f"Expanded query: {expanded_query}")
        
        return expanded_query