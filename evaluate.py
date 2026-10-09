from ingest import SessionLocal, DocumentChunk
from sentence_transformers import SentenceTransformer
from google import genai
import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from eval_dataset import GOLDEN_EVAL_DATASET
import time

load_dotenv()

# 1. Simulate pipeline
encoder = SentenceTransformer("all-MiniLM-L6-v2")
ai_client = genai.Client()

def run_rag_pipeline(question: str, top_k: int = 3) -> dict:
    """
    Simulates the core RAG pipeline without Redis cache
    """

    query_vector = encoder.encode(question).tolist()

    db = SessionLocal()
    try:
        results = (
            db.query(DocumentChunk)
            .order_by(DocumentChunk.embedding.cosine_distance(query_vector))
            .limit(top_k)
            .all()
        )

        retrieved_texts = [r.content for r in results]
    finally:
        db.close()

    # Propmt construction
    context_block = "\n\n---\n\n".join(retrieved_texts)
    prompt = f"""
    You are a precise engineering assistant. 
    Answer the user's question using ONLY the context provided below.
    If the context does not contain the answer, explicitly state: "I cannot answer this based on the provided documents."
            
    CONTEXT:
    {context_block}
            
    QUESTION: 
    {question}
    """

    response = ai_client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=prompt
    )

    return {
        "question": question,
        "contexts": retrieved_texts,
        "answer": response.text
    }

# 2. LLM judge
# Judge schema
class EvaluationResult(BaseModel):
    faithfulness_score: float = Field(
        description="Score between 0.0 and 1.0 indicating if all claims in the answer are supported by the context. Refusals for unanswerable questionds get 1.0"
    )
    faithfulness_reasoning: str = Field(
        description="Brief explanation of supported/unsupported claims."
    )
    context_precision_score: float = Field(
        description="Score between 0.0 and 1.0 measuring if the retrieved chunks directly contain the necessary information to answer the question."
    )
    context_precision_reasoning: str = Field(
        description="Explanation of chunk relevance."
    )

def judge_response(question: str, contexts: list[str], answer: str, ground_truth: str) -> EvaluationResult:
    judge_prompt = f"""
    You are an expert impartial judge evaluating a RAG pipeline. Evaluate the following answer based on the provided context and ground truth.

    QUESTION: {question}
    GROUND TRUTH: {ground_truth}
    RETRIEVED CONTEXTS:
    {chr(10).join(contexts)}

    GENERATED ANSWER:
    {answer}

    Rubric:
    1. Faithfulness: Is every factual claim in the generated answer directly inferable from the retrieved contexts?
                    If the answer correctly refuses to answer ('I cannot answer this based on the provided documents') when contexts lack information, score 1.0
    2. Context precision: Did the retrieved context contain the information needed to answer the question according to ground truth?
    """

    response = ai_client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=judge_prompt,
        config={
            'response_mime_type': 'application/json',
            'response_schema': EvaluationResult
        }
    )

    return response.parsed

# 3. Evaluation
def run_evaluation():
    print(f"\n Starting evaluation on {len(GOLDEN_EVAL_DATASET)} samples...\n")
    results = []

    for idx, item in enumerate(GOLDEN_EVAL_DATASET, 1):
        question = item["question"]
        ground_truth = item["ground_truth"]
        category = item["category"]

        print(f"[{idx}/{len(GOLDEN_EVAL_DATASET)}] Evaluating ({category}): {question}")

        # Run pipeline
        response = run_rag_pipeline(question)

        # Evaluate
        eval_result = judge_response(question, response["contexts"], response["answer"], ground_truth)

        results.append({
            "category": category,
            "question": question,
            "answer": response["answer"],
            "faithfulness": eval_result.faithfulness_score,
            "context_precision": eval_result.context_precision_score
        })

        time.sleep(1)

    print_report(results)

def print_report(results: list[dict]):
    # Calculate average
    total_faithfulness = sum(r["faithfulness"] for r in results)
    total_precision = sum(r["context_precision"] for r in results)
    
    avg_faithfulness = total_faithfulness / len(results)
    avg_precision = total_precision / len(results)

    # Per-category average
    category_counts = {}
    faithful_by_category = {}
    precision_by_category = {}

    for r in results:
        cat = r["category"]
        category_counts[cat] = category_counts.get(cat, 0) + 1
        faithful_by_category[cat] = faithful_by_category.get(cat, 0.0) + r["faithfulness"]
        precision_by_category[cat] = precision_by_category.get(cat, 0.0) + r["context_precision"]

    for cat in category_counts:
        faithful_by_category[cat] /= category_counts[cat]
        precision_by_category[cat] /= category_counts[cat]

    print("\n RAG EVALUATION REPORT")
    print("=" * 40)
    print(f"Overall Faithfulness: {avg_faithfulness:.3f}")
    print(f"Overall Context Precision: {avg_precision:.3f}")

    print("\n--- By Category ---")
    for cat in ["direct_fact", "paraphrase", "adversarial"]:
        print(f"{cat.upper()}")
        print(f"  Faithfulness:   {faithful_by_category[cat]:.3f}")
        print(f"  Context Precision: {precision_by_category[cat]:.3f}")

if __name__ == "__main__":
    run_evaluation()