#!/usr/bin/env python3
"""
Simple vLLM server for inference.

Usage:
    python finetune/vllm/vllm_server.py --model <model_path> --port <port>

Example:
    python finetune/vllm/vllm_server.py --model model/Qwen/Qwen3-0.6B --port 8000
"""

import argparse
import logging
from typing import List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from vllm import LLM, SamplingParams

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
llm_engine: Optional[LLM] = None
model_name: str = ""


class GenerateRequest(BaseModel):
    prompts: List[str]
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 512
    n: int = 1


class GenerateResponse(BaseModel):
    generated_texts: List[str]
    prompt_token_counts: List[int]
    completion_token_counts: List[int]


class HealthResponse(BaseModel):
    status: str
    model: str


def create_app(model_path: str, tensor_parallel_size: int = 1, gpu_memory_utilization: float = 0.9) -> FastAPI:
    """Create FastAPI app with vLLM engine."""
    global llm_engine, model_name
    
    app = FastAPI(title="Simple vLLM Server", version="1.0.0")
    
    # Initialize vLLM engine
    logger.info(f"Loading model: {model_path}")
    llm_engine = LLM(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        trust_remote_code=True,
    )
    model_name = model_path
    logger.info("Model loaded successfully")
    
    @app.get("/health", response_model=HealthResponse)
    async def health():
        """Health check endpoint."""
        return HealthResponse(status="ok", model=model_name)
    
    @app.post("/generate", response_model=GenerateResponse)
    async def generate(request: GenerateRequest):
        """Generate text from prompts."""
        if llm_engine is None:
            raise HTTPException(status_code=500, detail="Model not loaded")
        
        try:
            # Create sampling parameters
            sampling_params = SamplingParams(
                temperature=request.temperature,
                top_p=request.top_p,
                max_tokens=request.max_tokens,
                n=request.n,
            )
            
            # Generate outputs
            outputs = llm_engine.generate(request.prompts, sampling_params)
            
            # Extract results
            generated_texts = []
            prompt_token_counts = []
            completion_token_counts = []
            
            for output in outputs:
                generated_texts.extend([o.text for o in output.outputs])
                prompt_token_counts.extend([len(output.prompt_token_ids)] * len(output.outputs))
                completion_token_counts.extend([len(o.token_ids) for o in output.outputs])
            
            return GenerateResponse(
                generated_texts=generated_texts,
                prompt_token_counts=prompt_token_counts,
                completion_token_counts=completion_token_counts,
            )
            
        except Exception as e:
            logger.error(f"Generation error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    return app


def main():
    parser = argparse.ArgumentParser(description="Simple vLLM Server")
    parser.add_argument("--model", type=str, required=True, help="Model name or path")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9, help="GPU memory utilization")
    parser.add_argument("--log-level", type=str, default="info", help="Log level")
    
    args = parser.parse_args()
    
    # Create and run the app
    app = create_app(
        model_path=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()