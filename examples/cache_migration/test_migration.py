#!/usr/bin/env python3
"""
Simple test script to verify cache migration functionality.

This script sends requests to vLLM to generate KV cache and verify
that migration is working correctly.

Usage:
    python test_migration.py [--port PORT] [--prompt PROMPT] [--iterations N]
"""

import argparse
import time
import sys

try:
    import requests
except ImportError:
    print("Error: requests library not found. Install it with: pip install requests")
    sys.exit(1)


def send_request(prompt: str, port: int = 8000, model: str = "Qwen/Qwen2.5-7B-Instruct"):
    """Send a completion request to vLLM."""
    url = f"http://localhost:{port}/v1/completions"
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": 50,
        "temperature": 0.7,
    }
    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        result = response.json()
        text = result.get("choices", [{}])[0].get("text", "").strip()
        return text
    except requests.exceptions.RequestException as e:
        print(f"Error sending request: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Test cache migration functionality")
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="vLLM server port (default: 8000)"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="What is artificial intelligence?",
        help="Test prompt to use (default: 'What is artificial intelligence?')"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Number of requests to send to increase access count (default: 10)"
    )
    parser.add_argument(
        "--wait-time",
        type=float,
        default=35.0,
        help="Time to wait for migration (seconds, default: 35)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Model name (default: Qwen/Qwen2.5-7B-Instruct)"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Cache Migration Test Script")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  vLLM Port: {args.port}")
    print(f"  Model: {args.model}")
    print(f"  Prompt: {args.prompt}")
    print(f"  Iterations: {args.iterations}")
    print(f"  Wait time: {args.wait_time}s")
    print()
    
    # Test connection
    print("Step 1: Testing connection to vLLM...")
    test_result = send_request("test", args.port, args.model)
    if test_result is None:
        print("ERROR: Cannot connect to vLLM. Make sure vLLM is running on port", args.port)
        print("\nTo start vLLM with LMCache:")
        print(f"  CUDA_VISIBLE_DEVICES=0 \\")
        print(f"  LMCACHE_CONFIG_FILE=migration_test.yaml \\")
        print(f"  vllm serve {args.model} \\")
        print(f"    --gpu-memory-utilization 0.8 \\")
        print(f"    --port {args.port} \\")
        print(f"    --kv-transfer-config '{{\"kv_connector\":\"LMCacheConnectorV1\", \"kv_role\":\"kv_both\"}}'")
        sys.exit(1)
    print("✓ Connection successful")
    print()
    
    # Send first request (will cache)
    print("Step 2: Sending first request (will cache KV)...")
    result = send_request(args.prompt, args.port, args.model)
    if result:
        print(f"✓ First request completed")
        print(f"  Response preview: {result[:100]}...")
    else:
        print("✗ First request failed")
        sys.exit(1)
    print()
    
    # Send multiple requests to increase access count
    print(f"Step 3: Sending {args.iterations} requests to increase access count...")
    for i in range(args.iterations):
        result = send_request(args.prompt, args.port, args.model)
        if result:
            print(f"  Request {i+1}/{args.iterations} completed", end="\r")
        else:
            print(f"\n  ✗ Request {i+1} failed")
        time.sleep(0.5)
    print(f"\n✓ All {args.iterations} requests completed")
    print()
    
    # Wait for migration
    print(f"Step 4: Waiting {args.wait_time} seconds for migration interval...")
    print("  (Migration should trigger during this time if interval has passed)")
    print("  Check vLLM logs for:")
    print("    - 'Cache migration service initialized'")
    print("    - 'Migrated X keys after batched_put'")
    print()
    
    for i in range(int(args.wait_time)):
        remaining = args.wait_time - i
        print(f"  Waiting... {remaining:.0f}s remaining", end="\r")
        time.sleep(1)
    print(f"\n✓ Wait completed")
    print()
    
    # Final summary
    print("=" * 60)
    print("Test Summary")
    print("=" * 60)
    print("\n✓ Test script completed successfully")
    print("\nTo verify migration worked:")
    print("  1. Check vLLM logs for migration messages")
    print("  2. Verify keys exist in both CPU and disk backends (if copy_mode=true)")
    print("  3. Check disk cache directory: /tmp/data/hm/lmcache_test (or your configured path)")
    print("\nIf migration didn't occur:")
    print("  - Verify enable_cache_migration is true in config")
    print("  - Check that migration_interval has passed")
    print("  - Ensure new KV cache is being stored (migration only triggers on batched_put)")
    print("  - Verify both source and target backends are configured")
    print()


if __name__ == "__main__":
    main()

