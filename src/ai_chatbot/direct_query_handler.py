# src/ai_chatbot/direct_query_handler.py

import pandas as pd
import re
from typing import List, Dict, Any, Optional
from pathlib import Path


class DirectQueryHandler:
    """
    Handles direct CSV queries without LLM for fast response.
    Uses pandas DataFrame operations for instant results.
    """
    
    def __init__(self, csv_path: str = "db/defects_kedb.csv"):
        self.csv_path = Path(csv_path)
        self.df = None
        self._load_data()
    
    def _load_data(self):
        """Load CSV data into DataFrame."""
        try:
            if self.csv_path.exists():
                self.df = pd.read_csv(self.csv_path, keep_default_na=False)
                # Ensure column names are clean
                self.df.columns = self.df.columns.str.strip()
                print(f"[DirectQueryHandler] Loaded {len(self.df)} defects from CSV")
            else:
                print(f"[DirectQueryHandler] CSV not found at {self.csv_path}")
                self.df = pd.DataFrame()
        except Exception as e:
            print(f"[DirectQueryHandler] Error loading CSV: {e}")
            self.df = pd.DataFrame()
    
    def reload_data(self):
        """Reload data from CSV (useful after defect creation)."""
        self._load_data()
    
    def query(self, query_info: Dict[str, Any]) -> str:
        """
        Execute direct query on CSV data.
        
        Args:
            query_info: Dictionary containing extracted query information
                       from QueryClassifier
        
        Returns:
            Formatted string response
        """
        if self.df is None or self.df.empty:
            return "❌ No defect data available."
        
        # Start with all defects
        filtered_df = self.df.copy()
        
        # Debug: Print extracted filters
        print(f"[DirectQueryHandler] Extracted filters: {query_info}")
        print(f"[DirectQueryHandler] Starting with {len(filtered_df)} defects")
        
        # Apply filters based on extracted info
        if 'defect_ids' in query_info and query_info['defect_ids']:
            defect_ids = query_info['defect_ids']
            filtered_df = filtered_df[filtered_df['Issue key'].str.upper().isin(defect_ids)]
            print(f"[DirectQueryHandler] After defect_ids filter: {len(filtered_df)} defects")
        
        if 'priority' in query_info:
            priority = query_info['priority']
            filtered_df = filtered_df[filtered_df['Priority'].str.contains(
                priority, case=False, na=False)]
            print(f"[DirectQueryHandler] After priority filter ({priority}): {len(filtered_df)} defects")
        
        if 'severity' in query_info:
            severity = query_info['severity']
            filtered_df = filtered_df[filtered_df['Custom field (Severity)'].str.contains(
                severity, case=False, na=False)]
            print(f"[DirectQueryHandler] After severity filter ({severity}): {len(filtered_df)} defects")
        
        if 'release' in query_info:
            release = query_info['release']
            filtered_df = filtered_df[filtered_df['Custom field (MTP Gate)'].str.contains(
                release, case=False, na=False)]
            print(f"[DirectQueryHandler] After release filter ({release}): {len(filtered_df)} defects")
        
        if 'opco' in query_info:
            opco = str(query_info['opco']).strip().upper()
            opco_mask = None

            # Preferred explicit OpCo column names (if present in any dataset version)
            for opco_col in ['OpCo', 'OPCO', 'Opco', 'Operating Company']:
                if opco_col in filtered_df.columns:
                    opco_mask = filtered_df[opco_col].astype(str).str.strip().str.upper().eq(opco)
                    break

            # Fallback: infer from text fields when explicit OpCo column is unavailable.
            if opco_mask is None:
                summary = filtered_df['Summary'].astype(str) if 'Summary' in filtered_df.columns else pd.Series('', index=filtered_df.index)
                description = filtered_df['Description'].astype(str) if 'Description' in filtered_df.columns else pd.Series('', index=filtered_df.index)
                comments = filtered_df['Comments'].astype(str) if 'Comments' in filtered_df.columns else pd.Series('', index=filtered_df.index)
                text_blob = (summary + ' ' + description + ' ' + comments).str.upper()
                opco_mask = text_blob.str.contains(rf'\b{re.escape(opco)}\b', regex=True, na=False)

            if bool(opco_mask.any()):
                filtered_df = filtered_df[opco_mask]
                print(f"[DirectQueryHandler] After opco filter ({opco}): {len(filtered_df)} defects")
            else:
                # Data source has no explicit OpCo values; keep existing filtered set.
                query_info['_opco_unavailable'] = opco
                print(f"[DirectQueryHandler] OpCo '{opco}' not present in dataset columns/content; skipping OpCo filter")
        if 'status' in query_info:
            status = str(query_info['status']).strip().lower()
            status_series = filtered_df['Status'].astype(str).str.strip().str.lower()

            # Use exact match semantics to avoid treating "Reopened" as "Open".
            if status == 'open':
                # Treat Reopened as open for operational "open defects" views.
                mask = status_series.isin(['open', 'reopened'])
            elif status == 'closed':
                mask = status_series.eq('closed')
            elif status == 'resolved':
                mask = status_series.eq('resolved')
            elif status == 'new':
                mask = status_series.eq('new')
            elif status == 'in progress':
                mask = status_series.eq('in progress')
            elif status == 'reopened':
                mask = status_series.eq('reopened')
            else:
                mask = status_series.str.contains(status, na=False)

            filtered_df = filtered_df[mask]
            print(f"[DirectQueryHandler] After status filter ({query_info['status']}): {len(filtered_df)} defects")
        
        if 'assignee' in query_info:
            assignee = query_info['assignee']
            filtered_df = filtered_df[filtered_df['Assignee'].str.upper().str.contains(
                assignee, case=False, na=False)]
            print(f"[DirectQueryHandler] After assignee filter ({assignee}): {len(filtered_df)} defects")
        
        if 'reporter' in query_info:
            reporter = query_info['reporter']
            filtered_df = filtered_df[filtered_df['Reporter'].str.upper().str.contains(
                reporter, case=False, na=False)]
            print(f"[DirectQueryHandler] After reporter filter ({reporter}): {len(filtered_df)} defects")
        
        if 'tcode' in query_info:
            # Search in Summary, Description, and Comments for TCode
            tcode = query_info['tcode']
            tcode_filter = (
                filtered_df['Summary'].str.contains(tcode, case=False, na=False) |
                filtered_df['Description'].str.contains(tcode, case=False, na=False) |
                filtered_df['Comments'].str.contains(tcode, case=False, na=False)
            )
            filtered_df = filtered_df[tcode_filter]
            print(f"[DirectQueryHandler] After tcode filter ({tcode}): {len(filtered_df)} defects")
        
        # Date filtering
        if 'date_month' in query_info and 'date_year' in query_info:
            month = query_info['date_month']
            year = query_info['date_year']
            # Parse Created date and filter by month/year
            filtered_df = self._filter_by_month_year(filtered_df, month, year)
            print(f"[DirectQueryHandler] After date filter (month={month}, year={year}): {len(filtered_df)} defects")
        
        if 'date_after' in query_info:
            date_after = query_info['date_after']
            filtered_df = self._filter_by_date_after(filtered_df, date_after)
            print(f"[DirectQueryHandler] After date_after filter ({date_after}): {len(filtered_df)} defects")
        
        if 'date_before' in query_info:
            date_before = query_info['date_before']
            filtered_df = self._filter_by_date_before(filtered_df, date_before)
            print(f"[DirectQueryHandler] After date_before filter ({date_before}): {len(filtered_df)} defects")
        
        # Handle empty results
        if filtered_df.empty:
            return self._format_no_results(query_info)
        
        # Format response based on requested field
        if 'requested_field' in query_info:
            return self._format_field_response(filtered_df, query_info)
        else:
            return self._format_list_response(filtered_df, query_info)
    
    def _filter_by_month_year(self, df: pd.DataFrame, month: int, year: int) -> pd.DataFrame:
        """Filter DataFrame by month and year."""
        from datetime import datetime
        
        def match_month_year(date_str):
            try:
                # Parse date in DD/MM/YYYY HH:MM format
                date_obj = pd.to_datetime(date_str, format='%d/%m/%Y %H:%M', errors='coerce')
                if pd.isna(date_obj):
                    return False
                return date_obj.month == month and date_obj.year == year
            except:
                return False
        
        mask = df['Created'].apply(match_month_year)
        return df[mask]
    
    def _filter_by_date_after(self, df: pd.DataFrame, date_after: str) -> pd.DataFrame:
        """Filter DataFrame by date after."""
        from datetime import datetime
        
        try:
            # Parse the target date
            target_date = pd.to_datetime(date_after, format='%d/%m/%Y', errors='coerce')
            if pd.isna(target_date):
                return df
            
            def is_after(date_str):
                try:
                    date_obj = pd.to_datetime(date_str, format='%d/%m/%Y %H:%M', errors='coerce')
                    if pd.isna(date_obj):
                        return False
                    return date_obj >= target_date
                except:
                    return False
            
            mask = df['Created'].apply(is_after)
            return df[mask]
        except:
            return df
    
    def _filter_by_date_before(self, df: pd.DataFrame, date_before: str) -> pd.DataFrame:
        """Filter DataFrame by date before."""
        from datetime import datetime
        
        try:
            # Parse the target date
            target_date = pd.to_datetime(date_before, format='%d/%m/%Y', errors='coerce')
            if pd.isna(target_date):
                return df
            
            def is_before(date_str):
                try:
                    date_obj = pd.to_datetime(date_str, format='%d/%m/%Y %H:%M', errors='coerce')
                    if pd.isna(date_obj):
                        return False
                    return date_obj <= target_date
                except:
                    return False
            
            mask = df['Created'].apply(is_before)
            return df[mask]
        except:
            return df
    
    def _format_no_results(self, query_info: Dict[str, Any]) -> str:
        """Format response when no results found."""
        filters = []
        if 'priority' in query_info:
            filters.append(f"priority {query_info['priority']}")
        if 'severity' in query_info:
            filters.append(f"severity {query_info['severity']}")
        if 'release' in query_info:
            filters.append(f"release {query_info['release']}")
        if 'status' in query_info:
            filters.append(f"status {query_info['status']}")
        if 'opco' in query_info:
            filters.append(f"opco {query_info['opco']}")
        if 'assignee' in query_info:
            filters.append(f"assignee {query_info['assignee']}")
        if 'reporter' in query_info:
            filters.append(f"reporter {query_info['reporter']}")
        if 'tcode' in query_info:
            filters.append(f"TCode {query_info['tcode']}")
        if 'date_month' in query_info and 'date_year' in query_info:
            filters.append(f"created in {query_info['date_month']}/{query_info['date_year']}")
        
        if filters:
            msg = f"❌ No defects found matching: {', '.join(filters)}"
        else:
            msg = "❌ No defects found matching your query."

        if query_info.get('_opco_unavailable'):
            msg += (
                f"\nℹ️ Note: OpCo filter `{query_info['_opco_unavailable']}` could not be applied because "
                "this dataset has no OpCo field values."
            )
        return msg
    
    def _format_field_response(self, df: pd.DataFrame, query_info: Dict[str, Any]) -> str:
        """Format response when specific field is requested."""
        field = query_info['requested_field']
        
        if field == 'RootCause' or field == 'Resolution':
            return self._extract_from_comments(df, field, query_info)
        elif field == 'ImpactedTestCases' or field == 'ImpactedTestSets':
            return self._extract_from_comments(df, field, query_info)
        elif field == 'TCode':
            return self._extract_tcode(df, query_info)
        else:
            # Generic field extraction
            return self._format_list_response(df, query_info)
    
    def _extract_from_comments(self, df: pd.DataFrame, field: str, query_info: Dict[str, Any]) -> str:
        """Extract specific field from Comments column."""
        results = []
        
        for idx, row in df.iterrows():
            issue_key = row.get('Issue key', 'N/A')
            summary = row.get('Summary', 'N/A')
            comments = str(row.get('Comments', ''))
            
            # Extract the requested field from comments
            extracted_value = self._parse_field_from_comments(comments, field)
            
            if extracted_value:
                results.append({
                    'issue_key': issue_key,
                    'summary': summary,
                    'value': extracted_value
                })
        
        if not results:
            return f"❌ No {field} information found for the matching defects."
        
        # Format response
        response_parts = [f"**{field} Information:**\n"]
        
        for item in results[:10]:  # Limit to 10 results
            response_parts.append(
                f"\n**{item['issue_key']}** - {item['summary'][:60]}\n"
                f"{field}: {item['value']}\n"
            )
        
        if len(results) > 10:
            response_parts.append(f"\n... and {len(results) - 10} more results")
        
        return "".join(response_parts)
    
    def _parse_field_from_comments(self, comments: str, field: str) -> str:
        """Parse specific field from comments text."""
        if not comments:
            return "Not specified"
        
        # Define patterns for each field
        patterns = {
            'RootCause': [
                r'\*\*Root Cause:\*\*\s*(.+?)(?=\n\*\*|\n\n|$)',
                r'Root Cause:\s*(.+?)(?=\n|Resolution:|$)',
                r'(?:Root cause|Cause):\s*(.+?)(?=\n|$)'
            ],
            'Resolution': [
                r'\*\*Resolution:\*\*\s*(.+?)(?=\n\*\*|\n\n|$)',
                r'Resolution:\s*(.+?)(?=\n|Root Cause:|$)',
                r'(?:Resolution|Fix|Solution):\s*(.+?)(?=\n|$)'
            ],
            'ImpactedTestCases': [
                r'\*\*Impacted Test Cases:\*\*\s*(.+?)(?=\n\*\*|\n\n|$)',
                r'Impacted Test Cases:\s*(.+?)(?=\n|$)',
                r'Test Cases?:\s*(.+?)(?=\n|$)'
            ],
            'ImpactedTestSets': [
                r'\*\*Impacted Test Sets:\*\*\s*(.+?)(?=\n\*\*|\n\n|$)',
                r'Impacted Test Sets:\s*(.+?)(?=\n|$)',
                r'Test Sets?:\s*(.+?)(?=\n|$)'
            ]
        }
        
        if field not in patterns:
            return "Not specified"
        
        for pattern in patterns[field]:
            match = re.search(pattern, comments, re.IGNORECASE | re.DOTALL)
            if match:
                value = match.group(1).strip()
                # Clean up the value
                value = re.sub(r'\s+', ' ', value)
                return value[:200]  # Limit length
        
        return "Not specified"
    
    def _extract_tcode(self, df: pd.DataFrame, query_info: Dict[str, Any]) -> str:
        """Extract TCode information from defects."""
        results = []
        
        for idx, row in df.iterrows():
            issue_key = row.get('Issue key', 'N/A')
            summary = row.get('Summary', 'N/A')
            
            # Search for TCode pattern
            tcode = self._find_tcode_in_row(row)
            
            if tcode:
                results.append({
                    'issue_key': issue_key,
                    'summary': summary,
                    'tcode': tcode
                })
        
        if not results:
            return "❌ No TCode information found for the matching defects."
        
        # Format response
        response_parts = ["**TCode Information:**\n"]
        
        for item in results[:10]:
            response_parts.append(
                f"\n**{item['issue_key']}** - TCode: **{item['tcode']}**\n"
                f"Summary: {item['summary'][:60]}\n"
            )
        
        if len(results) > 10:
            response_parts.append(f"\n... and {len(results) - 10} more results")
        
        return "".join(response_parts)
    
    def _find_tcode_in_row(self, row: pd.Series) -> Optional[str]:
        """Find TCode in a defect row."""
        # Search in Summary, Description
        text = f"{row.get('Summary', '')} {row.get('Description', '')}"

        # Pattern for TCode: 2-4 uppercase letters followed by 1-3 digits
        pattern = r'\b([A-Z]{2,4}\d{1,3}[A-Z]?)\b'
        matches = re.findall(pattern, text)

        if matches:
            return matches[0]
        return None

    def keyword_search(self, query: str, top_k: int = 5) -> str:
        """
        Deterministic keyword fallback for descriptive queries when LLM is unavailable.
        """
        if self.df is None or self.df.empty:
            return ""

        keywords = self._extract_query_keywords(query)
        if not keywords:
            return ""

        scored_rows = []
        for _, row in self.df.iterrows():
            blob = " ".join([
                str(row.get('Summary', '')),
                str(row.get('Description', '')),
                str(row.get('Comments', '')),
            ]).lower()
            score = sum(1 for kw in keywords if kw in blob)
            if score > 0:
                scored_rows.append((score, row))

        if not scored_rows:
            return ""

        scored_rows.sort(key=lambda x: x[0], reverse=True)
        lines = [f"**Closest matching defects ({min(top_k, len(scored_rows))}):**\n"]
        for score, row in scored_rows[:top_k]:
            issue_key = str(row.get('Issue key', 'N/A'))
            summary = str(row.get('Summary', 'N/A'))
            status = str(row.get('Status', 'N/A'))
            severity = str(row.get('Custom field (Severity)', 'N/A'))
            lines.append(
                f"**{issue_key}** (match score: {score})\n"
                f"Summary: {summary[:120]}\n"
                f"Status: {status} | Severity: {severity}\n"
            )
        return "\n".join(lines).strip()

    def _extract_query_keywords(self, query: str) -> List[str]:
        q = (query or "").lower()
        stopwords = {
            'show', 'me', 'list', 'find', 'get', 'give', 'about', 'tell', 'explain', 'summarize',
            'defect', 'defects', 'issue', 'issues', 'the', 'a', 'an', 'and', 'or', 'for', 'with',
            'from', 'in', 'on', 'to', 'of', 'by', 'what', 'is', 'are', 'these', 'those', 'please',
            'suggest', 'solution', 'compare', 'difference', 'between'
        }
        words = re.findall(r"\b[a-zA-Z]{4,}\b", q)
        return [w for w in words if w not in stopwords]

        
        if matches:
            return matches[0]
        return None
    
    def _format_list_response(self, df: pd.DataFrame, query_info: Dict[str, Any]) -> str:
        """Format response as a list of defects."""
        # Limit to reasonable number
        display_df = df.head(20)
        
        response_parts = [f"**Found {len(df)} defect(s):**\n"]
        if query_info.get('_opco_unavailable'):
            response_parts.append(
                f"ℹ️ OpCo filter `{query_info['_opco_unavailable']}` was requested, but OpCo values are not present in this dataset; showing results based on other filters.\n"
            )
        
        for idx, row in display_df.iterrows():
            issue_key = str(row.get('Issue key', 'N/A'))
            summary = str(row.get('Summary', 'N/A'))
            status = str(row.get('Status', 'N/A'))
            severity = str(row.get('Custom field (Severity)', 'N/A'))
            assignee = str(row.get('Assignee', 'N/A'))
            
            # Find TCode if available
            tcode = self._find_tcode_in_row(row)
            tcode_str = f"| TCode: {tcode}" if tcode else ""
            
            response_parts.append(
                f"\n**{issue_key}** {tcode_str}\n"
                f"Summary: {summary[:80]}...\n"
                f"Status: {status} | Severity: {severity} | Assignee: {assignee}\n"
            )
        
        if len(df) > 20:
            response_parts.append(f"\n... and {len(df) - 20} more defects")
        
        return "".join(response_parts)
    
    def get_defect_details(self, defect_id: str) -> Optional[Dict[str, Any]]:
        """Get full details of a specific defect."""
        if self.df is None or self.df.empty:
            return None
        
        defect = self.df[self.df['Issue key'].str.upper() == defect_id.upper()]
        
        if defect.empty:
            return None
        
        row = defect.iloc[0]
        
        # Parse all fields
        comments = str(row.get('Comments', ''))
        
        return {
            'IssueKey': row.get('Issue key', 'N/A'),
            'Summary': row.get('Summary', 'N/A'),
            'Description': row.get('Description', 'N/A'),
            'Status': row.get('Status', 'N/A'),
            'Severity': row.get('Custom field (Severity)', 'N/A'),
            'Priority': row.get('Priority', 'N/A'),
            'Assignee': row.get('Assignee', 'N/A'),
            'Reporter': row.get('Reporter', 'N/A'),
            'Created': row.get('Created', 'N/A'),
            'Updated': row.get('Updated', 'N/A'),
            'Release': row.get('Custom field (MTP Gate)', 'N/A'),
            'TCode': self._find_tcode_in_row(row) or 'N/A',
            'RootCause': self._parse_field_from_comments(comments, 'RootCause'),
            'Resolution': self._parse_field_from_comments(comments, 'Resolution'),
            'ImpactedTestCases': self._parse_field_from_comments(comments, 'ImpactedTestCases'),
            'ImpactedTestSets': self._parse_field_from_comments(comments, 'ImpactedTestSets'),
            'Attachment': row.get('Attachment', 'None'),
            'Comments': comments
        }
    
    def format_defect_details(self, defect: Dict[str, Any]) -> str:
        """Format defect details for display."""
        return f"""
**Defect ID:** {defect['IssueKey']}
**Title:** {defect['Summary']}
**Status:** {defect['Status']} | **Severity:** {defect['Severity']} | **Priority:** {defect['Priority']}
**Assignee:** {defect['Assignee']} | **Reporter:** {defect['Reporter']}
**Created:** {defect['Created']} | **Updated:** {defect['Updated']}

**TCode:** {defect['TCode']}
**Release:** {defect['Release']}

**Description:**
{defect['Description']}

**Root Cause:**
{defect['RootCause']}

**Resolution:**
{defect['Resolution']}

**Impacted Test Cases:**
{defect['ImpactedTestCases']}

**Impacted Test Sets:**
{defect['ImpactedTestSets']}

**Attachment:** {defect['Attachment']}
        """.strip()