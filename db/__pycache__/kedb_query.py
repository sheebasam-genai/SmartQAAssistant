import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

CHROMA_DIR = "kedb_chroma"
COLLECTION_NAME = "defects_kedb"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class KEDBClient:
    def __init__(self):
        self.client = chromadb.Client(
            Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory=CHROMA_DIR,
            )
        )
        self.collection = self.client.get_collection(COLLECTION_NAME)
        self.model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    def search(self, query: str, top_k: int = 5):
        q_emb = self.model.encode([query]).tolist()[0]
        results = self.collection.query(
            query_embeddings=[q_emb],
            n_results=top_k,
        )
        # Normalize results for easier use
        hits = []
        for i in range(len(results["ids"][0])):
            hits.append(
                {
                    "id": results["ids"][0][i],
                    "document": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i]
                    if "distances" in results
                    else None,
                }
            )
        return hits


if __name__ == "__main__":
    kedb = KEDBClient()
    q = "login error 500 when user signs in"
    hits = kedb.search(q, top_k=3)
    for h in hits:
        print("----")
        print(f"DefectID: {h['metadata'].get('DefectID')}")
        print(f"Summary: {h['metadata'].get('Summary')}")
        print(f"Severity: {h['metadata'].get('Severity')}")
        print(h["document"])
        