#!/usr/bin/env python3
"""
Build ChromaDB database from defects_kedb.csv (Updated for new CSV format)

Key improvements:
1. Fixed CSV column mapping for new structure (Summary, Issue key, Custom field (Defect ID), Comments)
2. Added Comments parsing for RootCause, Resolution, ImpactedTestSets, ImpactedTestCases
3. Added TCode extraction from Summary/Description
4. Robust fallback handling for missing fields
5. Enhanced validation and logging
"""

import chromadb
import csv
import re
from pathlib import Path
from typing import Dict, List, Any
from sentence_transformers import SentenceTransformer
import shutil

# Configuration - Update these paths to match your setup
CSV_PATH = "defects_kedb.csv"  # Use uploaded file name
DB_PATH = "./chroma_db"
COLLECTION_NAME = "defects_kedb"

def extract_tcode_from_text(text: str) -> str:
    """Extract TCode from Summary or Description using common patterns"""
    tcode_patterns = [
        r'\b(ME\d{2}[MN]?|MM\d{2}[MN]?|FB\d{2}|VF\d{2}|MIRO|MB\d{2})\b',
        r'transaction\s+(\w+)',
        r'\b([A-Z]{2,4}\d{2}[A-Z]?)\b'
    ]
    
    text_upper = text.upper()
    for pattern in tcode_patterns:
        matches = re.findall(pattern, text_upper)
        if matches:
            return matches[0]
    return ""

def parse_comments_for_details(comments: str) -> Dict[str, str]:
    """Parse multiline Comments field for RootCause, Resolution, etc."""
    details = {
        'RootCause': '',
        'Resolution': '',
        'ImpactedTestSets': '',
        'ImpactedTestCases': ''
    }
    
    if not comments:
        return details
    
    lines = comments.split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Look for keywords indicating RootCause
        if re.search(r'root\s*cause|caused\s*by|due\s*to|issue\s*(is|was)', line, re.IGNORECASE):
            details['RootCause'] = line
        # Look for keywords indicating Resolution
        elif re.search(r'fix|resolved|solved|deployed|fixed|solution|workaround', line, re.IGNORECASE):
            details['Resolution'] = line
        # Look for test sets/cases
        elif re.search(r'test\s*(set|case)s?', line, re.IGNORECASE):
            if 'set' in line.lower():
                details['ImpactedTestSets'] = line
            else:
                details['ImpactedTestCases'] = line
    
    return details

def parse_defects_csv(csv_path: str) -> List[Dict[str, Any]]:
    """Parse the new CSV format with proper column mapping"""
    defects = []
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        for row_num, row in enumerate(reader, 1):
            if not (row.get('Issue key') or '').strip():
                continue
                
            issue_key = (row.get('Issue key') or '').strip()
            if not issue_key:
                continue
            
            defect = {
                'IssueKey': issue_key,
                'DefectID': (row.get('Custom field (Defect ID)') or issue_key).strip(),
                'Summary': (row.get('Summary') or '').strip(),
                'IssueType': (row.get('Issue Type') or 'Bug').strip(),
                'Status': (row.get('Status') or '').strip(),
                'Project': (row.get('Project key') or '').strip(),
                'Assignee': (row.get('Assignee') or '').strip(),
                'Reporter': (row.get('Reporter') or '').strip(),
                'Created': (row.get('Created') or '').strip(),
                'Updated': (row.get('Updated') or '').strip(),
                'Description': (row.get('Description') or '').strip(),
                'Attachment': (row.get('Attachment') or '').strip(),
                'Comments': (row.get('Comments') or '').strip(),
                'MTPGate': (row.get('Custom field (MTP Gate)') or '').strip(),
                'Severity': (row.get('Custom field (Severity)') or '').strip(),
                'Priority': (row.get('Priority') or '').strip(),
                
                # Parsed fields
                'Title': '',
                'RootCause': '',
                'Resolution': '',
                'TCode': '',
                'ImpactedTestSets': '',
                'ImpactedTestCases': ''
            }
            
            # Set Title from Summary if available
            if defect['Summary']:
                defect['Title'] = defect['Summary']
            
            # Extract TCode from Summary + Description
            tcode_text = f"{defect['Summary']} {defect['Description']}"
            defect['TCode'] = extract_tcode_from_text(tcode_text)
            
            # Parse Comments for additional details
            if defect['Comments']:
                comments_details = parse_comments_for_details(defect['Comments'])
                defect.update(comments_details)
            
            defects.append(defect)
            print(f"Parsed defect {len(defects)}: {issue_key}")
    
    print(f"✅ Parsed {len(defects)} defects from CSV")
    return defects

def create_document_text(defect: Dict[str, Any]) -> str:
    """Create comprehensive document text for semantic search"""
    return f"""IssueKey: {defect['IssueKey']}
DefectID: {defect['DefectID']}
Title: {defect['Title']}
Summary: {defect['Summary']}
Description: {defect['Description']}
TCode: {defect['TCode']}
Status: {defect['Status']}
Severity: {defect['Severity']}
Priority: {defect['Priority']}
Assignee: {defect['Assignee']}
Reporter: {defect['Reporter']}
RootCause: {defect['RootCause']}
Resolution: {defect['Resolution']}
ImpactedTestSets: {defect['ImpactedTestSets']}
ImpactedTestCases: {defect['ImpactedTestCases']}
Comments: {defect['Comments'][:500]}..."""  # Truncate long comments

def create_metadata(defect: Dict[str, Any]) -> Dict[str, str]:
    """Create metadata for filtering - all values as strings"""
    return {
        'IssueKey': str(defect['IssueKey']),
        'DefectID': str(defect['DefectID']),
        'Title': str(defect['Title'])[:100],  # ChromaDB metadata limit
        'Status': str(defect['Status']),
        'Severity': str(defect['Severity']),
        'Priority': str(defect['Priority']),
        'Assignee': str(defect['Assignee']),
        'Reporter': str(defect['Reporter']),
        'TCode': str(defect['TCode']),
        'Project': str(defect['Project'])
    }

def build_database():
    """Main function to build ChromaDB with complete data"""
    print("\n" + "="*80)
    print("🔄 BUILDING CHROMADB WITH COMPLETE CSV DATA")
    print("="*80 + "\n")
    
    # Step 1: Parse CSV with new format
    print("📊 [1/5] Parsing defects_kedb.csv...")
    defects = parse_defects_csv(CSV_PATH)
    
    if not defects:
        print("❌ No defects found in CSV!")
        return None
    
    # Step 2: Create ChromaDB
    print(f"🗄️  [2/5] Creating database at: {DB_PATH}")
    
    # Clean existing database
    if Path(DB_PATH).exists():
        shutil.rmtree(DB_PATH)
        print("   🗑️  Removed existing database")
    
    client = chromadb.PersistentClient(path=DB_PATH)
    
    # Create fresh collection
    try:
        client.delete_collection(COLLECTION_NAME)
    except:
        pass
    
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )
    
    # Step 3: Add all documents
    print("📝 [3/5] Adding documents to ChromaDB...")
    batch_size = 50
    
    for i in range(0, len(defects), batch_size):
        batch = defects[i:i+batch_size]
        
        ids = [defect['IssueKey'] for defect in batch]
        documents = [create_document_text(d) for d in batch]
        metadatas = [create_metadata(d) for d in batch]
        
        collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )
        print(f"   Added {i+len(batch)}/{len(defects)} defects", end='\r')
    
    print(f"\n✅ [4/5] Added {len(defects)} defects to collection")
    
    # Step 4: Validate COTEST-0010 specifically
    print("🔍 [5/5] Validating COTEST-0010...")
    results = collection.get(
        ids=["COTEST-0010"],
        include=['metadatas', 'documents']
    )
    
    if results['ids']:
        meta = results['metadatas'][0]
        doc = results['documents'][0]
        print(f"✅ COTEST-0010 VALIDATED:")
        print(f"   Title: {meta.get('Title', 'N/A')}")
        print(f"   TCode: {meta.get('TCode', 'N/A')}")
        print(f"   Status: {meta.get('Status', 'N/A')}")
        print(f"   Severity: {meta.get('Severity', 'N/A')}")
        print(f"   Document preview: {doc[:200]}...")
    else:
        print("❌ COTEST-0010 NOT FOUND!")
    
    print("\n" + "="*80)
    print("🎉 DATABASE BUILD COMPLETE!")
    print(f"📍 Location: {DB_PATH}")
    print(f"📦 Collection: {COLLECTION_NAME}")
    print(f"📊 Total defects: {collection.count()}")
    print("="*80)
    
    return collection

if __name__ == '__main__':
    collection = build_database()
    
    if collection:
        print("\n🚀 Ready to query! Run your chatbot.py now.")