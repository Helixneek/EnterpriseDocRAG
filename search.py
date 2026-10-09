from ingest import SessionLocal
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sentence_transformers import SentenceTransformer
from ingest import DocumentChunk, Base # Import from ingest.py

# Setup
DB_URL = "postgresql://dev_user:dev_password@localhost:5432/intelligence_db"
engine = create_engine(DB_URL)
SessionLocal = sessionmaker(bind=engine)

print("Loading model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# 2. Vector db search
def search_documents(user_query: str, top_k: int = 3):
    print(f"Searching for: '{user_query}'")

    # Step 1: Embed the user query into a vector
    query_vector = model.encode(user_query).tolist()

    db = SessionLocal()
    try:
        # Step 2: Actual search query
        results = (
            db.query(DocumentChunk)
            .order_by(DocumentChunk.embedding.cosine_distance(query_vector))
            .limit(top_k)
            .all()
        )

        # Step 3: display result
        print("\nTOP RESULTS:")
        for rank, chunk in enumerate(results, 1):
            print(f"\n[{rank}] Source: {chunk.source_file}")
            print(f"Content: {chunk.content.strip()}")

            # Exact distance score
            # distance = db.query(DocumentChunk.embedding.cosine_distance(query_vector)).filter(DocumentChunk.id == chunk.id).scalar()
            # print(f"Distance Score: {distance:.4f}")

    except Exception as e:
        print(f"Search failed: {e}")
    finally:
        db.close()

# Run test
if __name__ == "__main__":
    test_question = "How do vector databases store information"
    search_documents(test_question)