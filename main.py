import hashlib
import json
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import redis.asyncio as redis
from sentence_transformers import SentenceTransformer
from google import genai
from sse_starlette.sse import EventSourceResponse
import asyncio
from dotenv import load_dotenv

load_dotenv()

# Reuse prior SQLAlchemy setup from ingest
from ingest import SessionLocal, DocumentChunk

# Global var
ml_models = {}
redis_client = None

# Lifespan manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load embedding model into RAM on bootup
    print("Loading embedding model into memory...")
    ml_models["encoder"] = SentenceTransformer("all-MiniLM-L6-v2")

    # Connect to Redis
    global redis_client
    redis_host = os.getenv("REDIS_HOST", "localhost")
    redis_client = redis.Redis(host=redis_host, port=6379, db=0, decode_responses=True)

    yield

    # On shutdown
    print("Shutting down, clearing memory...")
    ml_models.clear()
    await redis_client.aclose()

app = FastAPI(lifespan=lifespan)

# Data validation
class QueryRequest(BaseModel):
    query: str
    top_k: int = 3

# Search endpoint
@app.post("/search")
async def search_api(request: QueryRequest):
    # Check the cache
    query_hash = hashlib.md5(request.query.strip().lower().encode()).hexdigest()
    cache_key = f"query:{query_hash}:{request.top_k}"

    cached_result = await redis_client.get(cache_key)

    if cached_result:
        print("Found something in cache! Returning instant response from RAM.")
        return {"source": "redis_cache", "results": json.loads(cached_result)}

    print("No result in cache! Running neural network and database search.")

    # Run the ML model
    query_vector = ml_models["encoder"].encode(request.query).tolist()

    # Database vector search
    db = SessionLocal()
    try:
        results = (
            db.query(DocumentChunk)
            .order_by(DocumentChunk.embedding.cosine_distance(query_vector))
            .limit(request.top_k)
            .all()
        )

        # Format results
        response_data = [
            {"source_file": r.source_file, "content": r.content}
            for r in results
        ]

        # Save cache for other users
        await redis_client.setex(
            cache_key,
            3600,
            json.dumps(response_data)
        )

        return {"source": "postgres_vector_db", "results": response_data}

    except Exception as e:
        raise HTTPException(status_code = 500, detail=str(e))
    finally:
        db.close()

@app.post("/search/stream")
async def search_stream_api(request: QueryRequest):
    # Check cache first
    query_hash = hashlib.md5(request.query.strip().lower().encode()).hexdigest()
    cache_key = f"rag_stream:{query_hash}:{request.top_k}"

    async def generate_stream():
        db = SessionLocal()

        try:
            # Scenario 1: Answer found in cache
            cached_result = await redis_client.get(cache_key)
            if cached_result:
                print("Found something in cache! Returning instant response from RAM.")
                parsed_cache = json.loads(cached_result)

                # Simulate streaming the cached answer
                words = parsed_cache["answer"].split(" ")
                for word in words:
                    yield f"data: {json.dumps({'chunk': word + ' '})}\n\n"
                    await asyncio.sleep(0.02)

                yield f"{json.dumps({'sources': parsed_cache['sources_used']})}\n\n"
                yield "[DONE]\n\n"
                return
        
            # Scenario 2: None in cache, run the full retrieval pipeline
            print("No result in cache! Running neural network and database search.")

            # Vector retrieval
            query_vector = ml_models["encoder"].encode(request.query).tolist()
            results = (
                db.query(DocumentChunk)
                .order_by(DocumentChunk.embedding.cosine_distance(query_vector))
                .limit(request.top_k)
                .all()
            )

            if not results:
                yield f"{json.dumps({'chunk': 'No relevant documents found in '})}"
                yield "[DONE]\n\n"
                return

            retrieved_texts = [r.content for r in results]
            sources = list(set([r.source_file for r in results]))

            # Propmt construction
            context_block = "\n\n---\n\n".join(retrieved_texts)
            prompt = f"""
            You are a precise engineering assistant. 
            Answer the user's question using ONLY the context provided below.
            If the context does not contain the answer, explicitly state: "I cannot answer this based on the provided documents."
            
            CONTEXT:
            {context_block}
            
            QUESTION: 
            {request.query}
            """

            # LLM generation
            #generation_model = genai.GenerativeModel('gemini-1.5-flash')
            #response_stream = generation_model.generate_content(prompt, stream=True)
            client = genai.Client()
            response_stream = client.models.generate_content_stream(model='gemini-3.5-flash-lite', contents=prompt)

            full_answer = ""

            # Stream response to client
            for chunk in response_stream:
                if chunk.text:
                    full_answer += chunk.text
                    yield f"{json.dumps({'chunk': chunk.text})}\n\n"

            # Cache the results
            yield f"data: {json.dumps({'sources': sources})}\n\n"
            yield "data: [DONE]\n\n"

            final_payload = {
                "query": request.query,
                "answer": full_answer,
                "sources_used": sources
            }
            await redis_client.setex(cache_key, 3600, json.dumps(final_payload))

        except Exception as e:
            yield f"{json.dumps({'error': str(e)})}\n\n"
        finally:
            db.close()
    
    return EventSourceResponse(generate_stream())