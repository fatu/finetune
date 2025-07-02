#!/usr/bin/env python3
"""
Simple vLLM client to communicate with vllm_server.py

Usage:
    python finetune/vllm/vllm_client.py --url http://localhost:8000 --prompt "Hello world"

Example:
    python finetune/vllm/vllm_client.py --url http://localhost:8000 --prompt "What is AI?"
"""

import argparse
import json
import sys
from typing import List, Optional

import requests


class VLLMClient:
    """Simple client for vLLM server."""
    
    def __init__(self, base_url: str, timeout: int = 30):
        """Initialize client with server URL."""
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
    
    def health_check(self) -> dict:
        """Check server health."""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Failed to connect to server: {e}")
    
    def generate(
        self,
        prompts: List[str],
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 512,
        n: int = 1
    ) -> dict:
        """Generate text from prompts."""
        payload = {
            "prompts": prompts,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "n": n
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/generate",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Failed to generate text: {e}")


def main():
    parser = argparse.ArgumentParser(description="Simple vLLM Client")
    parser.add_argument("--url", type=str, default="http://localhost:8000", help="Server URL")
    parser.add_argument("--prompt", type=str, help="Single prompt to generate from")
    parser.add_argument("--prompts", type=str, nargs="+", help="Multiple prompts to generate from")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p sampling parameter")
    parser.add_argument("--max-tokens", type=int, default=512, help="Maximum tokens to generate")
    parser.add_argument("--n", type=int, default=1, help="Number of completions per prompt")
    parser.add_argument("--timeout", type=int, default=30, help="Request timeout in seconds")
    parser.add_argument("--check-health", action="store_true", help="Only check server health")
    
    args = parser.parse_args()
    
    # Initialize client
    client = VLLMClient(args.url, timeout=args.timeout)
    
    try:
        # Health check
        if args.check_health:
            health = client.health_check()
            print("Server Health:")
            print(json.dumps(health, indent=2))
            return
        
        # Prepare prompts
        if args.prompt:
            prompts = [args.prompt]
        elif args.prompts:
            prompts = args.prompts
        else:
            print("Error: Please provide --prompt or --prompts")
            sys.exit(1)
        
        # Check server health first
        print("Checking server health...")
        health = client.health_check()
        print(f" Server is healthy. Model: {health.get('model', 'unknown')}")
        
        # Generate text
        print(f"\nGenerating text for {len(prompts)} prompt(s)...")
        result = client.generate(
            prompts=prompts,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            n=args.n
        )
        
        # Display results
        print("\nGeneration Results:")
        print("=" * 50)
        
        generated_texts = result.get("generated_texts", [])
        prompt_tokens = result.get("prompt_token_counts", [])
        completion_tokens = result.get("completion_token_counts", [])
        
        for i, (prompt, text) in enumerate(zip(prompts, generated_texts)):
            print(f"\nPrompt {i+1}: {prompt}")
            print(f"Generated: {text}")
            if i < len(prompt_tokens):
                print(f"Prompt tokens: {prompt_tokens[i]}")
            if i < len(completion_tokens):
                print(f"Completion tokens: {completion_tokens[i]}")
            print("-" * 50)
    
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()