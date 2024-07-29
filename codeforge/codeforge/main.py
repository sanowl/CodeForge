from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel
import logging
from codeforge.orchestrator.execution_pipeline import ExecutionPipeline
from codeforge.nlu.intent_classifier import IntentClassifier
from codeforge.code_intelligence.code_generator import CodeGenerator
from codeforge.context_management.project_indexer import ProjectIndexer
from codeforge.tool_integration.api_gateway import APIGateway
from codeforge.security.access_control import AccessControl

app = FastAPI()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CodeRequest(BaseModel):
    request: str

class CodeForge:
    def __init__(self):
        self.execution_pipeline = ExecutionPipeline()
        self.intent_classifier = IntentClassifier()
        self.code_generator = CodeGenerator()
        self.project_indexer = ProjectIndexer()
        self.api_gateway = APIGateway()
        self.access_control = AccessControl()

    async def process_request(self, user_id: str, request: str):
        logger.info(f"Processing request from user {user_id}: {request}")
        
        if not await self.access_control.validate_user(user_id):
            logger.warning(f"Access denied for user {user_id}")
            raise HTTPException(status_code=403, detail="Access denied")

        intent = await self.intent_classifier.classify(request)
        context = await self.project_indexer.get_context(user_id)
        
        result = await self.execution_pipeline.execute(intent, context, self.code_generator)
        
        await self.project_indexer.update_context(user_id, result)
        
        return result

codeforge = CodeForge()

@app.post("/generate")
async def generate_code(request: CodeRequest, token: str = Depends(oauth2_scheme)):
    user_id = await codeforge.access_control.get_user_id(token)
    return await codeforge.process_request(user_id, request.request)

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)