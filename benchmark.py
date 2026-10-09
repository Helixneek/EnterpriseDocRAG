import urllib.request
import json
import time
import uuid

API_URL = "http://127.0.0.1:8000/search"

def measure_latency(query: str) -> tuple[dict, float]:
    payload = json.dumps({"query": query, "top_k": 3}).encode("utf-8")

    req = urllib.request.Request(
        API_URL,
        data=payload,
        headers={"Content-Type": "application/json"}
    )

    start_time = time.perf_counter()
    with urllib.request.urlopen(req) as response:
        data = json.loads(response.read().decode("utf-8"))
    elapsed_ms = (time.perf_counter() - start_time) * 1000

    return data, elapsed_ms

def run_benchmark():
    # Use a unique query to guarantee cache miss on first run
    benchmark_query = f"What is the required encryption standard? [id:{uuid.uuid4().hex[:6]}]"

    print("\nRunning latency benchmark...\n")

    # 1. First call: miss (will be slow, has to manually grab from database) (Embedding + pgvector SQL query)
    data_1, latency_1 = measure_latency(benchmark_query)
    print(f"[First call - Cache miss]: {latency_1:.2f} ms")

    # 2. Second call: hit with same query (instant, returns cached answer in RAM) (Redis RAM lookup)
    data_2, latency_2 = measure_latency(benchmark_query)
    print(f"[Second call - Cache hit]: {latency_2:.2f} ms")

    # Calculate speedup factor
    speedup = latency_1 / latency_2
    print(f"Speedup factor: {speedup:.2f}x")
    pass

if __name__ == "__main__":
    run_benchmark()
