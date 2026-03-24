"""
FastAPI Web Interface for SalesOS Agent
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from datetime import datetime
from pydantic import BaseModel
from typing import Optional 
import json
import uvicorn
import asyncio
import uuid
from pathlib import Path
from contextlib import asynccontextmanager

from utils.logger import setup_logging, get_logger

# Import agent
from agent import create_sales_agent, ask_agent, stream_agent
from config import REQUEST_TIMEOUT


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    
    setup_logging()
    logger = get_logger(__name__)

    logger.info("Initializing agent...")
    try:
        app.state.agent = create_sales_agent()
       
        logger.info("Agent ready")
    except Exception as e:
        
        logger.exception("Failed to initialize agent")
        app.state.agent = None
    
    yield

    logger.info("Shutting down")
   

app = FastAPI(
    title="Sales Agent API",
    description="AI assistant with sales data and knowledge base capabilities",
    version="0.2",
    lifespan=lifespan
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger = get_logger(__name__)

    logger.error(
        f"Unhandled exception on {request.url}",
        exc_info=True
    )

    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger = get_logger(__name__)

    logger.info(f"{request.method} {request.url.path} started")

    try:
        response = await call_next(request)
        logger.info(
            f"{request.method} {request.url.path} completed "
            f"with status {response.status_code}"
        )
        return response

    except Exception:
        logger.exception(
            f"{request.method} {request.url.path} failed"
        )
        raise


# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8000", "http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class QuestionRequest(BaseModel):
    question: str
    thread_id: Optional[str] = None
    

class QuestionResponse(BaseModel):
    answer: str
    thread_id: str
    timestamp: str


class HealthResponse(BaseModel):
    status: str
    agent_ready: bool


# API Endpoints
@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the web interface"""
    html_file = Path(__file__).parent / "static" / "index.html"
    if html_file.exists():
        return html_file.read_text(encoding="utf-8")
    else:
        return """
        <html>
            <body>
                <h1>Sales Agent API</h1>
                <p>API is running. Create static/index.html for the web interface.</p>
                <p>Or use the API directly:</p>
                <ul>
                <li>GET /health - Check API status</li>
                <li>POST /ask - Ask the agent a question</li>
                <li>POST /ask/stream - Ask the agent with streaming response</li>
                <li>GET /tools - List available tools</li>
                <li>GET /docs - Interactive API documentation</li>
                </ul>
            </body>
        </html>
        """


@app.get("/health", response_model=HealthResponse)
async def health_check(request: Request):
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        agent_ready=hasattr(request.app.state, "agent") and request.app.state.agent is not None
    )


@app.post("/ask", response_model=QuestionResponse)
async def ask_question(request_data: QuestionRequest, request: Request):
    """
    Ask the agent a question
    
    Example:
    ```
    POST /ask
    {
        "question": "What were our top customers in Q1?",
        "thread_id": "user123"
    }
    ```
    """
    agent = request.app.state.agent
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    # Ensure a thread_id exists
    thread_id = request_data.thread_id or str(uuid.uuid4())

    try:
        loop = asyncio.get_running_loop()
        
        # Add timeout to prevent hanging requests
        answer = await asyncio.wait_for(
            loop.run_in_executor(
                None,
                ask_agent,
                agent,
                request_data.question,
                thread_id,
                False
            ),
            timeout=REQUEST_TIMEOUT
        )

        return QuestionResponse(
            answer=answer,
            thread_id=thread_id,
            timestamp=datetime.now().isoformat()
        )
        
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=504,
            detail=f"Request timed out after {REQUEST_TIMEOUT} seconds. Please try a simpler question."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@app.post("/ask/stream")
async def ask_question_stream(request_data: QuestionRequest, request: Request):
    # TODO: Requires further updates to properly stream responses, currently uses typewriter effect
    """
    Ask the agent a question with streaming response (Server-Sent Events)
    
    Example:
    ```
    POST /ask/stream
    {
        "question": "What were our top customers in Q1?",
        "thread_id": "user123"
    }
    ```
    
    Response format:
    ```
    data: {"type": "content", "data": "..."}
    data: {"type": "done", "thread_id": "user123"}
    ```
    """
    agent = request.app.state.agent
    
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    thread_id = request_data.thread_id or str(uuid.uuid4())

    async def event_generator():
        start_time = asyncio.get_running_loop().time()

        try:
            # Run stream_agent in executor since it's synchronous
            loop = asyncio.get_running_loop()
            
            # Create a queue for thread-safe communication
            queue = asyncio.Queue()
            
            def run_stream():
                try:
                    for chunk in stream_agent(agent, request_data.question, thread_id):
                        # Put each chunk in the queue
                        asyncio.run_coroutine_threadsafe(queue.put(chunk), loop)
                except Exception as e:
                    try:
                        asyncio.run_coroutine_threadsafe(
                            queue.put({"error": str(e)}), loop
                        )
                    except RuntimeError:
                        pass
                finally:
                    # Signal completion
                    asyncio.run_coroutine_threadsafe(queue.put(None), loop)
            
            # Start streaming in background thread
            future = loop.run_in_executor(None, run_stream)
            
            # Yield chunks as they arrive
            while True:
                if await request.is_disconnected():
                    break

                if asyncio.get_running_loop().time() - start_time > REQUEST_TIMEOUT:
                    yield f"data: {json.dumps({'type': 'error', 'message': 'Streaming timeout'})}\n\n"
                    break

                chunk = await queue.get()
                
                if chunk is None:
                    # Stream complete
                    yield f"data: {json.dumps({'type': 'done', 'thread_id': thread_id})}\n\n"
                    break
                
                if isinstance(chunk, dict) and "error" in chunk:
                    yield f"data: {json.dumps({'type': 'error', 'message': chunk['error']})}\n\n"
                    break
                    
                # Yield the content chunk
                yield f"data: {json.dumps({'type': 'content', 'data': chunk})}\n\n"
                
            if not future.done():
                future.cancel()

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # Disable nginx buffering
        }
    )


@app.get("/tools")
async def list_tools(request: Request):
    """List available tools"""
    agent = request.app.state.agent
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    return {
        "tools": [
            {
                "name": "query_sales_database",
                "description": "Query sales data using natural language",
                "examples": [
                    "What were total sales in Q1?",
                    "Who are our top 5 customers?"
                ]
            },
            {
                "name": "search_local_docs",
                "description": "Search company knowledge base",
                "examples": [
                    "What is our discount policy?",
                    "What are our Q1 sales goals?"
                ]
            },
            {
                "name": "wiki_summary",
                "description": "Get Wikipedia summaries",
                "examples": [
                    "Who is Elon Musk?",
                    "What is quantum computing?"
                ]
            },
            {
                "name": "create_chart",
                "description": "Create interactive visualizations",
                "examples": [
                    "Show me a bar chart of top customers",
                    "Create a line chart of monthly revenue"
                ]
            },
            {
                "name": "create_multi_series_chart",
                "description": "Create multi-series charts for comparing metrics",
                "examples": [
                    "Compare sales vs targets by month",
                    "Visualize revenue vs expenses"
                ]
            }
        ]
    }


# Run with: uvicorn api:app --reload --host 127.0.0.1 --port 8000
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
