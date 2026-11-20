import os
import sys
import pprint

# Ensure project root is on sys.path so `src` imports work when running this file directly.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.repository.defect_repository import DefectRepository
from src.utils.rag_helper import retrieve_kedb_context

def main():
    repo = DefectRepository()
    test_query = "COTEST-0010"

    print(f"=== RAW retrieval for '{test_query}' ===")
    raw_results = retrieve_kedb_context(test_query, k=50) or []
    print(f"Raw results count: {len(raw_results)}")
    for i, item in enumerate(raw_results[:6]):  # show up to 6 items for brevity
        print(f"\nRaw[{i}] type: {type(item)}")
        pprint.pprint(item)

    print("\n\n=== Normalized results from DefectRepository.query_defects ===")
    normalized = repo.query_defects(test_query, k=20)
    print(f"Normalized results count: {len(normalized)}")

    if not normalized:
        print("No normalized results returned. Check raw results and parsing logic.")
    for i, defect in enumerate(normalized):
        print("\n" + "-"*40)
        print(f"Result #{i}")
        print(f"IssueKey: {defect.get('IssueKey')}")
        print(f"Title: {defect.get('Title')}")
        print(f"Summary: {defect.get('Summary')}")
        print(f"TCode: {defect.get('TCode')}")
        print(f"Status: {defect.get('Status')}")
        print("Full normalized record:")
        pprint.pprint(defect)

if __name__ == "__main__":
    main()