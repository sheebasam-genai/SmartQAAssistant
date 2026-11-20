# build_kedb_vectordb.py
# Adds project root to sys.path so src.* imports work when run from db/
# Uses sentence-transformers for embeddings and SimpleDoc wrapper for Chroma

import os
import sys
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import Chroma   # keep this if you have langchain_community installed
# If you prefer, you can use chromadb client directly instead of langchain_community

# Ensure project root is on sys.path so `from src.repository...` works
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.repository.defect_repository import parse_comments_for_details, extract_tcode_from_text

class SimpleDoc:
    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata
def safe_str(value):
        if value is None:
            return ""
        return str(value)
def build_vectordb(csv_path: str = r"C:\Users\Sheeba\GENAIPOC\SmartQAAssistant\db\defects_kedb.csv", chroma_path: str = "db/chroma_db", model_name: str = "all-MiniLM-L6-v2"):
    print("Loading KEDB from CSV...")
    df = pd.read_csv(csv_path)
    df = df.fillna("")

    documents = []
    metadatas = []

    print("Processing KEDB entries for vector database...")
    for _, row in tqdm(df.iterrows(), total=df.shape[0]):
        tcode = extract_tcode_from_text(row.get("Summary", "")) or extract_tcode_from_text(row.get("Description", ""))
        parsed_comments = parse_comments_for_details(row.get("Comments", ""))

        content = (
            f"Summary: {row.get('Summary','')}\n"
            f"Description: {row.get('Description','')}\n"
            f"Comments: {row.get('Comments','')}\n"
            f"TCode: {tcode}\n"
            f"Root Cause: {parsed_comments.get('RootCause', '')}\n"
            f"Resolution: {parsed_comments.get('Resolution', '')}\n"
            f"Impacted Test Sets: {parsed_comments.get('ImpactedTestSets', '')}\n"
            f"Impacted Test Cases: {parsed_comments.get('ImpactedTestCases', '')}"
        )

        documents.append(content)
        metadatas.append({
            "Issue key": safe_str(row.get("Issue key")),
            "Issue Type": safe_str(row.get("Issue Type")),
            "Status": safe_str(row.get("Status")),
            "Project key": safe_str(row.get("Project key")),
            "Assignee": safe_str(row.get("Assignee")),
            "Reporter": safe_str(row.get("Reporter")),
            "Created": safe_str(row.get("Created")),
            "Updated": safe_str(row.get("Updated")),
            "Attachment": safe_str(row.get("Attachment")),
            "DefectID": safe_str(row.get("Custom field (Defect ID)")),
            "Summary": safe_str(row.get("Summary")),
            "Description": safe_str(row.get("Description")),
            "Comments": safe_str(row.get("Comments")),
            "TCode": safe_str(tcode),
            "RootCause": safe_str(parsed_comments.get("RootCause")),
            "Resolution": safe_str(parsed_comments.get("Resolution")),
            "ImpactedTestSets": safe_str(parsed_comments.get("ImpactedTestSets")),
            "ImpactedTestCases": safe_str(parsed_comments.get("ImpactedTestCases")),
        })

    print("Initializing sentence-transformers model...")
    model = SentenceTransformer(model_name)

    print("Embedding documents...")
    # sentence-transformers returns numpy arrays; Chroma wrapper expects an embedding function or embeddings object.
    # We'll compute embeddings ourselves and pass Document-like objects to Chroma.from_documents with embedding param set to the numpy embeddings list.
    embeddings = [model.encode(doc, show_progress_bar=False) for doc in tqdm(documents)]

    print("Creating Chroma DB (using SimpleDoc wrapper)...")
    docs = [SimpleDoc(doc, meta) for doc, meta in zip(documents, metadatas)]

    # Note: langchain_community.vectorstores.Chroma.from_documents expects an "embedding" object,
    # but it may also accept precomputed embeddings depending on version. If from_documents doesn't accept the embeddings list,
    # you can either:
    #  - Use Chroma.from_documents with an embedding object (e.g., a wrapper class with embed_documents method), or
    #  - Use the chromadb client directly to add documents + embeddings.
    #
    # We'll try to call from_documents with the embeddings object (some versions accept SentenceTransformer embeddings as a simple object).
    try:
        vectordb = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,            # may or may not be accepted depending on version
            persist_directory=chroma_path
        )
    except Exception:
        # Fallback: use chromadb client directly to create a collection and upsert embeddings
        try:
            import chromadb
            client = chromadb.Client()
            collection_name = "kedb_collection"
            # Remove existing collection if present
            try:
                client.delete_collection(collection_name)
            except Exception:
                pass
            coll = client.create_collection(name=collection_name)
            # Use simple IDs
            ids = [str(i) for i in range(len(documents))]
            metadatas_list = metadatas
            documents_list = documents
            # embeddings must be list of lists (floats)
            coll.add(documents=documents_list, metadatas=metadatas_list, ids=ids, embeddings=embeddings)
            print(f"Chroma (chromadb) collection created: {collection_name}")
            return
        except Exception as e:
            print("Failed to create Chroma DB with both langchain_community and chromadb client:", e)
            raise

    try:
        vectordb.persist()
    except Exception:
        pass

    print(f"Chroma DB built successfully at {chroma_path}")

if __name__ == "__main__":
    build_vectordb()