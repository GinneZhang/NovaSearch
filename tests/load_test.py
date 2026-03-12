"""
Load Test for NovaSearch API.

Measures latency and throughput under concurrent requests.
"""

import os
import sys
import json
import time
import threading
import statistics
from typing import List, Dict

try:
    import httpx
except ImportError:
    httpx = None

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def single_request(base_url: str, headers: dict, query: str, results: list, idx: int):
    """Execute a single /ask request and record latency."""
    start = time.time()
    status = "error"
    answer_length = 0
    
    try:
        with httpx.Client(timeout=60) as client:
            resp = client.post(
                f"{base_url}/ask",
                json={"query": query, "top_k": 3},
                headers=headers
            )
            
            if resp.status_code == 200:
                status = "success"
                for line in resp.text.strip().split("\n"):
                    if line:
                        try:
                            data = json.loads(line)
                            if data.get("type") == "token":
                                answer_length += len(data.get("content", ""))
                        except json.JSONDecodeError:
                            pass
            else:
                status = f"http_{resp.status_code}"
    except Exception as e:
        status = f"exception: {str(e)[:50]}"
    
    latency = time.time() - start
    results[idx] = {
        "latency": round(latency, 3),
        "status": status,
        "answer_length": answer_length
    }


def run_load_test(
    num_concurrent: int = 10,
    query: str = "What happens if I violate the insider trading policy?"
):
    """
    Run a load test with N concurrent requests.
    """
    if not httpx:
        print("Error: httpx is required. Install with: pip install httpx")
        sys.exit(1)
    
    base_url = os.getenv("NOVASEARCH_URL", "http://localhost:8000")
    api_key = os.getenv("API_KEY", "")
    headers = {"X-API-KEY": api_key}
    
    print(f"\n{'=' * 60}")
    print(f"NovaSearch Load Test")
    print(f"{'=' * 60}")
    print(f"  Target:      {base_url}")
    print(f"  Concurrency: {num_concurrent}")
    print(f"  Query:       {query[:60]}...")
    print(f"{'=' * 60}\n")
    
    # Initialize results array
    results: List[Dict] = [None] * num_concurrent
    threads = []
    
    start_time = time.time()
    
    for i in range(num_concurrent):
        t = threading.Thread(
            target=single_request,
            args=(base_url, headers, query, results, i)
        )
        threads.append(t)
        t.start()
    
    for t in threads:
        t.join()
    
    total_time = time.time() - start_time
    
    # Analyze results
    latencies = [r["latency"] for r in results if r and r["status"] == "success"]
    successes = sum(1 for r in results if r and r["status"] == "success")
    failures = num_concurrent - successes
    
    print(f"Results:")
    print(f"  Total time:     {total_time:.2f}s")
    print(f"  Successes:      {successes}/{num_concurrent}")
    print(f"  Failures:       {failures}/{num_concurrent}")
    
    if latencies:
        print(f"\nLatency Statistics (successful requests):")
        print(f"  Min:            {min(latencies):.3f}s")
        print(f"  Max:            {max(latencies):.3f}s")
        print(f"  Mean:           {statistics.mean(latencies):.3f}s")
        print(f"  Median:         {statistics.median(latencies):.3f}s")
        if len(latencies) > 1:
            print(f"  Std Dev:        {statistics.stdev(latencies):.3f}s")
        print(f"  P95:            {sorted(latencies)[int(len(latencies) * 0.95)]:.3f}s")
        print(f"  Throughput:     {successes / total_time:.2f} req/s")
    
    print(f"\n{'=' * 60}")
    
    # Per-request detail
    print("\nPer-Request Detail:")
    for i, r in enumerate(results):
        if r:
            print(f"  [{i+1:2d}] {r['status']:>10s}  latency={r['latency']:.3f}s  chars={r['answer_length']}")
    
    return results


if __name__ == "__main__":
    concurrency = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    run_load_test(num_concurrent=concurrency)
