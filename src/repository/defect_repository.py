# src/repository/defect_repository.py
# ── UPDATED: Added attachment retrieval helpers (Feature 3)
import re
import json
from typing import List, Dict, Any, Optional
from pathlib import Path
import json
from typing import List, Dict, Any, Optional
from sentence_transformers import util, SentenceTransformer
import textwrap
from pathlib import Path
from src.utils.rag_helper import retrieve_kedb_context
import logging
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.DEBUG)
from src.utils.rag_helper import retrieve_kedb_context
import logging
import json
from pathlib import Path

logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.DEBUG)

# Updated FIELD_KEYWORDS to match new CSV structure
FIELD_KEYWORDS = {
    'DefectID': ['defect id', 'defectid', 'id', 'defect #', 'd#'],
    'Title': ['title', 'name', 'subject'],
    'DefectSummary': ['summary', 'defect summary', 'issue summary'],
    'Description': ['description', 'details', 'desc'],
    'TCode': ['tcode', 'transaction code', 'tc', 'trans code'],
    'Status': ['status', 'state'],
    'Assignee': ['assignee', 'assigned to', 'owner'],
    'Reporter': ['reporter', 'reported by', 'raised by'],
    'RootCause': ['root cause', 'cause', 'reason', 'why'],
    'Resolution': ['resolution', 'fix', 'solution', 'resolved'],
    'ImpactedTestSets': ['test sets', 'testset', 'ts', 'impacted test sets'],
    'ImpactedTestCases': ['test cases', 'testcase', 'tc', 'impacted test cases'],
    'MTPGate': ['mtp gate', 'gate'],
    'Severity': ['severity', 'defect severity'],
    'Priority': ['priority', 'defect priority'],
    'Attachment': ['attachment', 'attachments', 'files', 'uploaded files']
}

FIELD_KEY_TO_DEFECT_KEY = {
    'issue key': 'IssueKey', 'issuekey': 'Issuekey', 'key': 'Issue',
    'title': 'Title', 'name': 'Title', 'subject': 'Title',
    'summary': 'DefectSummary', 'defect summary': 'DefectSummary',
    'description': 'Description', 'details': 'Description', 'desc': 'Description',
    'tcode': 'TCode', 'transaction code': 'TCode', 'tc': 'TCode',
    'status': 'Status', 'state': 'Status',
    'assignee': 'Assignee', 'assigned to': 'Assignee', 'owner': 'Assignee',
    'reporter': 'Reporter', 'reported by': 'Reporter', 'raised by': 'Reporter',
    'root cause': 'RootCause', 'cause': 'RootCause', 'reason': 'RootCause',
    'resolution': 'Resolution', 'fix': 'Resolution', 'solution': 'Resolution',
    'test sets': 'ImpactedTestSets', 'testset': 'ImpactedTestSets', 'ts': 'ImpactedTestSets',
    'test cases': 'ImpactedTestCases', 'testcase': 'ImpactedTestCases',
    'mtp gate': 'MTPGate',
    'severity': 'Severity',
    'priority': 'Priority',
    'attachment': 'Attachment', 'attachments': 'Attachment'
}


def parse_comments_for_details(comments: str) -> Dict[str, str]:
    """Parse multiline Comments field to extract RootCause, Resolution, etc."""
    if not comments:
        return {}
    
    result = {
        'RootCause': '',
        'Resolution': '',
        'ImpactedTestSets': '',
        'ImpactedTestCases': ''
    }
    
    lines = comments.split('\n')
    current_section = None
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Root Cause section
        if re.search(r'root cause|cause|reason|why', line, re.IGNORECASE):
            current_section = 'RootCause'
            result['RootCause'] += line + ' '
        elif current_section == 'RootCause':
            result['RootCause'] += line + ' '
            
        # Resolution section  
        elif re.search(r'resolution|fix|solution|resolved', line, re.IGNORECASE):
            current_section = 'Resolution'
            result['Resolution'] += line + ' '
        elif current_section == 'Resolution':
            result['Resolution'] += line + ' '
            
        # Impacted Test Sets
        elif re.search(r'test sets?|ts?', line, re.IGNORECASE):
            result['ImpactedTestSets'] += line + ' '
            
        # Impacted Test Cases
        elif re.search(r'test cases?|tc', line, re.IGNORECASE) and 'ts' not in line.lower():
            result['ImpactedTestCases'] += line + ' '
    
    # Clean up
    for key in result:
        result[key] = re.sub(r'\s+', ' ', result[key].strip())
    
    return result


def _parse_kedb_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Parse KEDB CSV results into normalized defect format"""
    normalized_defects = []
    
    for row in results:
        # Map CSV columns to internal keys
        defect = {
            'Title': row.get('Summary', ''),
            'IssueKey': row.get('Issue key', ''),
            'IssueType': row.get('Issue Type', ''),
            'Status': row.get('Status', ''),
            'ProjectKey': row.get('Project key', ''),
            'Assignee': row.get('Assignee', ''),
            'Reporter': row.get('Reporter', ''),
            'Created': row.get('Created', ''),
            'Updated': row.get('Updated', ''),
            'Description': row.get('Description', ''),
            'Attachment': row.get('Attachment', ''),
            'DefectSummary': row.get('Summary', ''),
        }
        
        # Extract TCode
        tcode_sources = [
            defect['Title'], 
            defect['Description'], 
            defect['IssueKey']
        ]
        for source in tcode_sources:
            tcode = extract_tcode_from_text(source)
            if tcode:
                defect['TCode'] = tcode
                break
        
        # Parse Comments for additional fields
        parsed_comments = parse_comments_for_details(row.get('Comments', ''))
        defect.update(parsed_comments)
        
        normalized_defects.append(defect)
    
    return normalized_defects


def extract_tcode_from_text(text: str) -> Optional[str]:
    """Extract TCode from Summary, Description, or Issue key using patterns like MM03, ME23N"""
    if not text:
        return None
    
    # Common TCode patterns: MM*, ME*, MB*, etc. (2-4 letters + 1-4 digits)
    tcode_pattern = r'\b([A-Z]{2,4}\d{1,4})\b'
    matches = re.findall(tcode_pattern, text, re.IGNORECASE)
    return matches[0] if matches else None


class DefectRepository:
    def __init__(self, chroma_path: str = "db/chroma_db", 
                 attachments_path: str = "db/attachments.json"):
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.chroma_path = chroma_path
        self.attachments_path = Path(attachments_path)

    def extract_requested_fields(self, query: str) -> List[str]:
        q = (query or "").lower()
        requested = []
        for field_key, keywords in FIELD_KEYWORDS.items():
            for kw in keywords:
                try:
                    if re.search(kw, q):
                        requested.append(field_key)
                        break
                except re.error:
                    if kw in q:
                        requested.append(field_key)
                        break
        return ["all"] if "all" in requested else list(dict.fromkeys(requested))

    def extract_single_defect_id(self, query: str) -> str:
        """UPDATED: Now matches COTEST-0001 format"""
        q = (query or "").upper()
        # First try COTEST-XXXX format
        m = re.search(r"\b(COTEST-\d+)\b", q)
        if m:
            return m.group(1)
        # Fallback to other formats
        m = re.search(r"\b(SMAR-\d+|D\d{3})\b", q)
        return m.group(1) if m else ""

    def is_list_intent(self, query: str) -> bool:
        q = (query or "").lower()
        list_words = r"\b(show|list|find|get|search|give me|display|what are)\b"
        if re.search(list_words, q) and re.search(r"\bdefect(s)?\b", q):
            return True
        return False

    def wants_testcases_or_testsets(self, query: str):
        q = (query or "").lower()
        return {
            "testcases": bool(re.search(r"\b(test case|test cases?|tcs?|tc_|impacted test cases?)\b", q)),
            "testsets": bool(re.search(r"\b(test set|test sets?|impacted test sets?)\b", q)),
        }

    def indent_multiline(self, text: str, indent: int = 4) -> str:
        """Indent each line of the given multiline text by the specified number of spaces."""
        indentation = ' ' * indent
        return '\n'.join(indentation + line if line.strip() else line for line in text.splitlines())
    
    @staticmethod
    def matches_assignee(defect: dict, assignee_query: str) -> bool:
        assignee = defect.get("Assignee", "").lower()
        comments = defect.get("Comments", "").lower()
        return assignee_query.lower() in assignee or assignee_query.lower() in comments
    
    @staticmethod
    def extract_tcodes(defect: dict) -> set:
        text_fields = [
            defect.get("Summary", ""),
            defect.get("Description", ""),
            defect.get("Comments", "")
        ]
        tcodes = set()
        pattern = re.compile(r'\b([A-Z]{2}\d{1,3})\b', re.IGNORECASE)
        for text in text_fields:
            matches = pattern.findall(text)
            for m in matches:
                tcodes.add(m.upper())
        return tcodes
    
    @staticmethod
    def matches_tcode(defect: dict, tcode_query: str) -> bool:
        tcodes = DefectRepository.extract_tcodes(defect)
        return tcode_query.upper() in tcodes

    def format_full_record_markdown(self, defect: Dict[str, Any]) -> str:
        """Format complete defect record with parsed fields"""
        tcode = defect.get('TCode', 'N/A')
        comments = str(defect.get("Comments") or defect.get("comments") or "").strip()
        comments_block = comments if comments else "N/A"

        description = defect.get('Description') or defect.get('DefectSummary') or "N/A"
        description_block = self.indent_multiline(description)

        # Get attachment info
        attachments = self.get_attachments_for_defect(defect.get('IssueKey', ''))
        attachment_info = self._format_attachment_info(defect, attachments)

        body = f"""
    🐛 Defect: {defect.get('DefectID', 'N/A')} ({defect.get('IssueKey', 'N/A')})

    **Summary**: {defect.get('Title', defect.get('DefectSummary', 'N/A'))}

    **📋 Details**:
    **TCode**: `{tcode}`
    **Status**: {defect.get('Status', 'N/A')}
    **Assignee**: {defect.get('Assignee', 'N/A')}
    **Reporter**: {defect.get('Reporter', 'N/A')}
    **Created**: {defect.get('Created', 'N/A')}

    **📝 Description**:
    {description_block}

    **🔍 Analysis**:
    **Root Cause**: {defect.get('RootCause', 'Not specified')}
    **Resolution**: {defect.get('Resolution', 'Not specified')}
    **Impacted Test Sets**: {defect.get('ImpactedTestSets', 'Not specified')}
    **Impacted Test Cases**: {defect.get('ImpactedTestCases', 'Not specified')}

    **📎 Attachments**: {attachment_info}
    **Comments:**
    {comments_block}
    """
        return textwrap.dedent(body).strip()

    def _format_attachment_info(self, defect: Dict[str, Any], attachments: List[Dict]) -> str:
        """Format attachment information for display"""
        # Check CSV attachment field first
        csv_attachments = defect.get('Attachment', defect.get('attachments', ''))
        
        if attachments:
            attachment_names = [att['file_name'] for att in attachments]
            return ', '.join(attachment_names) + f" ({len(attachments)} file(s))"
        elif csv_attachments:
            return str(csv_attachments)
        else:
            return 'None'

    def get_attachments_for_defect(self, issue_key: str) -> List[Dict[str, Any]]:
        """
        Retrieve attachments for a given defect issue key.
        
        Args:
            issue_key: The defect issue key (e.g., 'COTEST-0001')
            
        Returns:
            List of attachment dictionaries with metadata (without base64 data for display)
        """
        try:
            if not self.attachments_path.exists():
                return []
            
            with open(self.attachments_path, 'r') as f:
                all_attachments = json.load(f)
            
            attachments = all_attachments.get(issue_key, [])
            # Return metadata without the large base64 data for listing purposes
            return [
                {
                    'file_name': att['file_name'],
                    'file_type': att['file_type'],
                    'file_size': att['file_size'],
                    'mime_type': att['mime_type'],
                    'uploaded_at': att.get('uploaded_at', 'Unknown')
                }
                for att in attachments
            ]
        except Exception as e:
            logger.error(f"Error loading attachments for {issue_key}: {e}")
            return []

    def get_attachment_content(self, issue_key: str, file_name: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve specific attachment content including base64 data.
        
        Args:
            issue_key: The defect issue key
            file_name: The attachment file name
            
        Returns:
            Dict with attachment data or None if not found
        """
        try:
            if not self.attachments_path.exists():
                return None
            
            with open(self.attachments_path, 'r') as f:
                all_attachments = json.load(f)
            
            attachments = all_attachments.get(issue_key, [])
            for att in attachments:
                if att['file_name'] == file_name:
                    return att
            return None
        except Exception as e:
            logger.error(f"Error loading attachment {file_name} for {issue_key}: {e}")
            return None

    def format_resolution_only(self, record: Dict[str, Any]) -> str:
        return f"""**{record.get('IssueKey', 'N/A')}**
        **Resolution:** {record.get('Resolution', 'Not available')}
        **RootCause:** {record.get('RootCause', 'Not identified')}"""

    def format_testcases_block(self, records: List[Dict[str, Any]]) -> str:
        all_tcs = set()
        for r in records:
            tcs = r.get("ImpactedTestCases", "")
            if tcs:
                all_tcs.update([tc.strip() for tc in str(tcs).split(",") if tc.strip()])
        
        if not all_tcs:
            return "**Impacted Test Cases:** None found in comments"
        return "**Impacted Test Cases:**\n" + "\n".join(f"- {tc}" for tc in sorted(all_tcs))
    
    def filter_defects(self, defects: List[Dict[str, Any]], filters: Dict[str, str]) -> List[Dict[str, Any]]:
        filtered = []
        for defect in defects:
            match = True
            for key, val in filters.items():
                defect_val = defect.get(key, "").lower()
                if val.lower() not in defect_val:
                    match = False
                    break
            if match:
                filtered.append(defect)
        return filtered
    
    @staticmethod
    def extract_filters_from_query(query: str) -> Dict[str, str]:
        filters = {}
        severity_match = re.search(r'\bP[1-5]\b', query, re.IGNORECASE)
        if severity_match:
            filters['Severity'] = severity_match.group(0).upper()
        release_match = re.search(r'release\s+([Rr]\d+(\.\d+)*)', query)
        if release_match:
            filters['Release'] = release_match.group(1)
        assignee_match = re.search(r'assigned to (\w+)', query, re.IGNORECASE)
        if assignee_match:
            filters['Assignee'] = assignee_match.group(1)
        return filters
    
    def query_defects(self, query: str, k: int = 10, mode="Hybrid", where: Dict[str, Any] = None):
        logger.debug(f"query_defects called with query: {query}")
        filters = DefectRepository.extract_filters_from_query(query)
        
        # Check for explicit IssueKey in query
        issue_keys = re.findall(r"\b(COTEST-\d+)\b", query.upper())
        if issue_keys:
            results = []
            for ik in issue_keys:
                logger.debug(f"Searching for explicit issue key: {ik}")
                kedb_results = retrieve_kedb_context(ik, k=50) or []
                logger.debug(f"retrieve_kedb_context returned {len(kedb_results)} results for {ik}")
                candidates = self._parse_kedb_results(kedb_results)
                exact_matches = [d for d in candidates if d.get("IssueKey", "").upper() == ik.upper()]
                logger.debug(f"Exact matches found: {len(exact_matches)} for {ik}")
                results.extend(exact_matches)
            return results[:k]
        
        kedb_results = retrieve_kedb_context(query, k=k*2) or []
        logger.debug(f"Semantic retrieve returned {len(kedb_results)} results")
        candidates = self._parse_kedb_results(kedb_results)
        filtered = self.filter_defects(candidates, filters)
        return filtered[:k]

    def _parse_kedb_results(self, kedb_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """UPDATED: Uses "Issue key" as primary identifier"""
        defects = []
        for idx, result in enumerate(kedb_results):
            if isinstance(result, dict):
                metadata = result.get('metadata', {}) or result
            else:
                metadata = {}

            issue_key = (metadata.get("Issue key") or "").strip()
            title = metadata.get("Summary", "").strip()
            summary = metadata.get("Summary", "").strip()
            description = metadata.get("Description", "").strip()
            comments = metadata.get("Comments", "").strip()

            tcode = extract_tcode_from_text(title + " " + description)
            comment_details = parse_comments_for_details(comments)

            status = metadata.get("Status", "Unknown")
            issue_type = metadata.get("Issue Type", "Bug")

            normalized = {
                "IssueKey": issue_key or "Unknown",
                "Title": title,
                "Summary": summary,
                "Description": description,
                "TCode": tcode,
                "Status": status,
                "IssueType": issue_type,
                "RootCause": comment_details.get("RootCause", ""),
                "Resolution": comment_details.get("Resolution", ""),
                "ImpactedTestSets": comment_details.get("ImpactedTestSets", ""),
                "ImpactedTestCases": comment_details.get("ImpactedTestCases", ""),
                "Release": metadata.get("Release", ""),
                "MTPGate": metadata.get("Custom field (MTP Gate)", ""),
                "Severity": metadata.get("Custom field (Severity)", ""),
                "Priority": metadata.get("Priority", ""),
                "Attachment": metadata.get("Attachment", "")
            }

            if issue_key or title:
                defects.append(normalized)

        return defects


    # ── NEW: Attachment retrieval helpers (Feature 3) ──
    def get_attachments_for_defect(self, issue_key: str) -> List[dict]:
        """Return list of attachment metadata dicts for a given defect."""
        att_path = Path("db/attachments.json")
        if not att_path.exists():
            return []
        try:
            with open(att_path, "r") as f:
                data = json.load(f)
            return data.get(issue_key, [])
        except (json.JSONDecodeError, FileNotFoundError):
            return []

    def get_attachment_content(self, issue_key: str, file_name: str) -> Optional[dict]:
        """Return a single attachment dict (with base64_content) by issue_key + file_name."""
        attachments = self.get_attachments_for_defect(issue_key)
        for att in attachments:
            if att.get("file_name") == file_name:
                return att
        return None

    # ── Feature 3: Attachment retrieval methods ──
    ATTACHMENTS_PATH = Path("db/attachments.json")

    def get_attachments_for_defect(self, issue_key: str) -> List[Dict[str, Any]]:
        """Return attachment metadata (without base64 content) for a defect."""
        if not self.ATTACHMENTS_PATH.exists():
            return []
        try:
            with open(self.ATTACHMENTS_PATH, 'r') as f:
                data = json.load(f)
            attachments = data.get(issue_key.upper(), [])
            # Return metadata only (strip large base64 content)
            return [
                {
                    "file_name": att.get("file_name", ""),
                    "file_size": att.get("file_size", 0),
                    "uploaded_at": att.get("uploaded_at", ""),
                }
                for att in attachments
            ]
        except Exception as e:
            print(f"[DefectRepository] Error reading attachments: {e}")
            return []

    def get_attachment_content(self, issue_key: str, file_name: str) -> Optional[Dict[str, Any]]:
        """Retrieve a specific attachment's full data (including base64 content)."""
        if not self.ATTACHMENTS_PATH.exists():
            return None
        try:
            with open(self.ATTACHMENTS_PATH, 'r') as f:
                data = json.load(f)
            for att in data.get(issue_key.upper(), []):
                if att.get("file_name") == file_name:
                    return att
        except Exception as e:
            print(f"[DefectRepository] Error reading attachment content: {e}")
        return None