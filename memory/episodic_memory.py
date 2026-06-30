"""
Episodic Memory Module
─────────────────────────────────────────────────────────────────
Stores past loan applications and their outcomes in a ChromaDB vector store.
Allows agents to query for semantically similar past cases to inform their
reasoning (e.g. Risk Agent comparing current applicant to past defaults).
"""

import os
import json
import chromadb
from chromadb.config import Settings

# Persistent storage directory
DB_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "chroma_db")
os.makedirs(DB_DIR, exist_ok=True)

# Initialize ChromaDB client
client = chromadb.PersistentClient(path=DB_DIR, settings=Settings(anonymized_telemetry=False))

# We use the default embedding function (all-MiniLM-L6-v2)
collection = client.get_or_create_collection(
    name="loan_episodes",
    metadata={"hnsw:space": "cosine"}
)

def _format_document(app_data: dict, credit_data: dict) -> str:
    """Creates a rich text representation of the applicant for embedding."""
    # We want the embedding to capture the core financial and demographic profile
    return (
        f"Age: {app_data.get('applicant_age')}, "
        f"Gender: {app_data.get('applicant_gender')}, "
        f"Location: {app_data.get('applicant_location')}, "
        f"Employment: {app_data.get('employment_type')}, "
        f"Income: {app_data.get('monthly_income')}, "
        f"Loan Amount: {app_data.get('loan_amount')}, "
        f"Credit Score: {credit_data.get('credit_score')}, "
        f"Defaults: {credit_data.get('total_defaults')}"
    )

def store_episode(application_id: str, app_data: dict, credit_data: dict, 
                  decision: str, risk_score: float, bias_detected: bool, 
                  debate_summary: str = "") -> None:
    """
    Embed and store a completed loan application in the vector database.
    """
    if not application_id or not app_data:
        return

    document = _format_document(app_data, credit_data or {})
    
    # Store key outcomes as metadata so they can be retrieved and filtered
    metadata = {
        "application_id": application_id,
        "decision": decision,
        "risk_score": float(risk_score) if risk_score is not None else -1.0,
        "bias_detected": bool(bias_detected),
        "debate_summary": debate_summary[:200] if debate_summary else "None"
    }

    try:
        collection.add(
            documents=[document],
            metadatas=[metadata],
            ids=[application_id]
        )
    except Exception as e:
        print(f"Failed to store episode in ChromaDB: {e}")

def retrieve_similar_cases(app_data: dict, credit_data: dict, top_k: int = 5) -> list[dict]:
    """
    Retrieve the most semantically similar past applications from the vector database.
    Returns a list of dictionaries containing the document and metadata of past cases.
    """
    if collection.count() == 0:
        return []

    document = _format_document(app_data, credit_data or {})

    try:
        results = collection.query(
            query_texts=[document],
            n_results=min(top_k, collection.count())
        )
        
        cases = []
        if results and results.get("documents") and results["documents"][0]:
            for i in range(len(results["documents"][0])):
                cases.append({
                    "id": results["ids"][0][i],
                    "profile": results["documents"][0][i],
                    "outcome": results["metadatas"][0][i]
                })
        return cases
    except Exception as e:
        print(f"Failed to retrieve similar cases: {e}")
        return []
