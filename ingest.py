import os
from sqlalchemy import text
from sqlalchemy import create_engine, Column, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import declarative_base, sessionmaker
from pgvector.sqlalchemy import Vector
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path

# 1. Database schema & connection
db_host = os.getenv("DB_HOST", "localhost")
DB_URL = f"postgresql://dev_user:dev_password@{db_host}:5432/intelligence_db"
engine = create_engine(DB_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id = Column(Integer, primary_key=True, index=True)
    source_file = Column(String, index=True)    # The original file
    content = Column(Text)                      # Actual text chunk contents
    metadata_json = Column(JSONB)               # Metadata like page number, author, etc
    embedding = Column(Vector(384))             # Mathematical representation

# 2. Embedding model
# Using a lightweight local model that outputs 384-d vectors
print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# 3. The chunker
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,       # Number of characters per chunk
    chunk_overlap = 50,     # Overlap to prevent cutting sentences early
    separators=["\n\n", "\n", ".", " ", ""] # Where to cut
)

# 4. Content ingestion pipeline
def process_and_store_document(file_name: str, raw_text: str):
    print(f"Processing {file_name}...")

    # Chunking
    chunks = text_splitter.split_text(raw_text)
    print(f"Split into {len(chunks)} chunks.")

    embeddings = model.encode(chunks)

    # Database insertion
    db = SessionLocal()
    try:
        # Delete existing records from the same file
        db.query(DocumentChunk).filter(DocumentChunk.source_file == file_name).delete()
        print(f"Deleted existing records for {file_name}.")

        # Start saving records
        db_records = []
        for i, (chunk_text, vector) in enumerate(zip(chunks, embeddings)):
            record = DocumentChunk(
                source_file = file_name,
                content = chunk_text,
                metadata_json = {"chunk_index": i},
                embedding = vector
            )
            db_records.append(record)

        db.bulk_save_objects(db_records)
        db.commit()
        print(f"Successfully saved {len(db_records)} records to Postgres DB.")

    except Exception as e:
        db.rollback()
        print(f"Database error: {e}")

    finally:
        db.close()

if __name__ == "__main__":
    # Create pgvector extension and tables if they dont exist
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        conn.commit()

    Base.metadata.create_all(bind=engine)

    file_path = Path(__file__).parent / "mock_text.txt"

    if not file_path.exists():
        raise FileNotFoundError(f"Source file not found at: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        text_content = f.read()

    process_and_store_document(file_path.name, text_content)