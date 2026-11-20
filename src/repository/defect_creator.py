# src/repository/defect_creator.py
# Enhanced with Attachment Support (Feature 3)

import pandas as pd
import re
import json
import base64
import os
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime

# ── Constants ──
CSV_FIELDNAMES = [
    'Summary', 'Issue key', 'Issue Type', 'Status', 'Project key', 
    'Assignee', 'Reporter', 'Created', 'Updated', 'Description', 
    'Attachment', 'Comments',
    'Custom field (MTP Gate)',
    'Custom field (Severity)',
    'Priority'
]

ALLOWED_FILE_TYPES = {
    'jpg', 'jpeg', 'png', 'gif',   # Images
    'txt', 'log',                    # Logs
    'pdf', 'doc', 'docx'            # Documents
}

MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB


# ── Feature 3: Attachment validation dataclass ──
@dataclass
class AttachmentValidationResult:
    """Result of validating a single attachment."""
    valid: bool
    file_name: str
    file_size: int = 0
    error: str = ""


@dataclass
class Attachment:
    """Represents a stored attachment."""
    file_name: str
    file_size: int
    content_base64: str
    uploaded_at: str = ""


# ── Validation helpers ──
def validate_file_type(file_name: str) -> bool:
    """Check if file extension is in allowed list."""
    ext = file_name.rsplit('.', 1)[-1].lower() if '.' in file_name else ''
    return ext in ALLOWED_FILE_TYPES


def validate_file_size(file_bytes: bytes) -> bool:
    """Check if file size is within the 5 MB limit."""
    return len(file_bytes) <= MAX_FILE_SIZE


def validate_attachment(file_bytes: bytes, file_name: str) -> AttachmentValidationResult:
    """Validate a single attachment (type + size)."""
    if not validate_file_type(file_name):
        ext = file_name.rsplit('.', 1)[-1].lower() if '.' in file_name else '(none)'
        return AttachmentValidationResult(
            valid=False, file_name=file_name,
            error=f"Unsupported file type '.{ext}'. Allowed: {', '.join(sorted(ALLOWED_FILE_TYPES))}"
        )
    if not validate_file_size(file_bytes):
        size_mb = len(file_bytes) / (1024 * 1024)
        return AttachmentValidationResult(
            valid=False, file_name=file_name, file_size=len(file_bytes),
            error=f"File too large ({size_mb:.1f} MB). Max allowed: 5 MB"
        )
    return AttachmentValidationResult(valid=True, file_name=file_name, file_size=len(file_bytes))


def validate_multiple_attachments(files: List[Tuple[bytes, str]]) -> List[AttachmentValidationResult]:
    """Validate a list of (file_bytes, file_name) tuples."""
    return [validate_attachment(fb, fn) for fb, fn in files]


def encode_attachment_to_base64(file_bytes: bytes) -> str:
    return base64.b64encode(file_bytes).decode('utf-8')


def decode_attachment_from_base64(b64_str: str) -> bytes:
    return base64.b64decode(b64_str)


# ── Existing helpers ──
def generate_new_issue_key(csv_path: str) -> str:
    """Generate next Issue Key (COTEST-0001, COTEST-0002, etc.) from CSV"""
    try:
        df = pd.read_csv(csv_path, keep_default_na=False)
        issue_keys = df['Issue key'].dropna().astype(str).tolist()
        issue_numbers = []
        for issue_key in issue_keys:
            match = re.match(r'COTEST-(\d+)', issue_key.upper())
            if match:
                issue_numbers.append(int(match.group(1)))
        next_issue_num = max(issue_numbers + [0]) + 1
        return f'COTEST-{next_issue_num:04d}'
    except:
        return 'COTEST-0051'


def build_comments_field(root_cause: str = '', resolution: str = '', 
                    impacted_test_sets: str = '', impacted_test_cases: str = '') -> str:
    """Build multiline Comments field for new defects"""
    comments_parts = []
    def is_valid(val):
        return val and str(val).lower() != 'nan'

    if is_valid(root_cause):
        comments_parts.append(f"**Root Cause:** {root_cause}")
    if is_valid(resolution):
        comments_parts.append(f"**Resolution:** {resolution}")
    if is_valid(impacted_test_sets):
        comments_parts.append(f"**Impacted Test Sets:** {impacted_test_sets}")
    if is_valid(impacted_test_cases):
        comments_parts.append(f"**Impacted Test Cases:** {impacted_test_cases}")
    return '\n'.join(comments_parts)


class DefectCreator:
    def __init__(self, csv_path: str = "db/defects_kedb.csv", chroma_path: str = "db/chroma_db"):
        self.csv_path = Path(csv_path)
        self.chroma_path = Path(chroma_path)
        self.attachments_path = Path("db/attachments.json")
        self.ensure_csv_exists()
        self._ensure_attachments_file()
    
    def ensure_csv_exists(self):
        if not self.csv_path.exists():
            df = pd.DataFrame(columns=CSV_FIELDNAMES)
            df.to_csv(self.csv_path, index=False)
    
    def _ensure_attachments_file(self):
        """Ensure db/attachments.json exists."""
        self.attachments_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.attachments_path.exists():
            with open(self.attachments_path, 'w') as f:
                json.dump({}, f)
    
    def _append_to_csv(self, row: Dict[str, Any]) -> bool:
        try:
            sanitized_row = {k: str(row.get(k, '')) for k in CSV_FIELDNAMES}
            df_new = pd.DataFrame([sanitized_row])
            df_new.to_csv(self.csv_path, mode='a', header=False, index=False, encoding='utf-8')
            return True
        except Exception as e:
            import traceback
            print("Exception while writing to CSV:")
            traceback.print_exc()
            return False
    
    def _store_attachments(self, issue_key: str, files: List[Tuple[bytes, str]]) -> List[str]:
        """
        Validate, encode, and store attachments in db/attachments.json.
        Returns list of stored file names.
        """
        if not files:
            return []

        stored_names = []
        # Load existing attachments
        with open(self.attachments_path, 'r') as f:
            all_attachments = json.load(f)

        attachment_list = []
        for file_bytes, file_name in files:
            result = validate_attachment(file_bytes, file_name)
            if result.valid:
                b64 = encode_attachment_to_base64(file_bytes)
                attachment_list.append({
                    "file_name": file_name,
                    "file_size": len(file_bytes),
                    "content_base64": b64,
                    "uploaded_at": datetime.now().strftime('%d/%m/%Y %H:%M')
                })
                stored_names.append(file_name)
            else:
                print(f"[DefectCreator] Skipping invalid attachment: {result.error}")

        if attachment_list:
            all_attachments[issue_key] = attachment_list
            with open(self.attachments_path, 'w') as f:
                json.dump(all_attachments, f, indent=2)

        return stored_names
    
    def create_defect(self, defect_data: Dict[str, Any],
                      attachments: Optional[List[Tuple[bytes, str]]] = None) -> Tuple[bool, str, str]:
        """
        Create new defect with updated CSV format.
        
        Args:
            defect_data: Dictionary with defect fields
            attachments: Optional list of (file_bytes, file_name) tuples (Feature 3)
        
        Returns:
            (success: bool, message: str, issue_key: str)
        """
        issue_key = generate_new_issue_key(str(self.csv_path))

        # Store attachments if provided
        stored_file_names = []
        if attachments:
            stored_file_names = self._store_attachments(issue_key, attachments)

        attachment_field = ', '.join(stored_file_names) if stored_file_names else defect_data.get('attachment', '')

        csv_row = {
            'Summary': defect_data.get('Title', ''), 
            'Issue key': issue_key,
            'Issue Type': 'Bug',
            'Status': 'New',
            'Project key': 'COTEST',
            'Assignee': defect_data.get('assignee', ''),
            'Reporter': defect_data.get('reporter', 'SYSTEM'),
            'Created': pd.Timestamp.now().strftime('%d/%m/%Y %H:%M'),
            'Updated': pd.Timestamp.now().strftime('%d/%m/%Y %H:%M'),
            'Description': defect_data.get('Description', ''),
            'Attachment': attachment_field,
            'Comments': build_comments_field(
                root_cause='Unknown',
                resolution='Under investigation.',
                impacted_test_sets=defect_data.get('ImpactedTestSets', ''),
                impacted_test_cases=defect_data.get('ImpactedTestCases', '')
            ),
            'Custom field (Severity)': defect_data.get('Severity', ''),
            'Custom field (OpCo)': defect_data.get('OpCo', ''),
            'Custom field (Release)': defect_data.get('Release', ''),
            'Custom field (TCode)': defect_data.get('TCode', '')
        }
        
        if self._append_to_csv(csv_row):
            att_msg = ""
            if stored_file_names:
                att_msg = f" with {len(stored_file_names)} attachment(s)"
            message = f"New defect created successfully{att_msg}! Issue Key: {issue_key}"
            return True, message, issue_key
        else:
            return False, "Failed to save defect to CSV file", ""
