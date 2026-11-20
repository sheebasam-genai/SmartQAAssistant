# src/ai_chatbot/query_classifier.py
# ── UPDATED: Added PATTERN_ANALYSIS query type recognition (Feature 4)

import re
from typing import Dict, List, Any


class QueryClassifier:
    """
    Smart query classifier that routes queries to:
    - DIRECT: Direct CSV lookup (fast) - for data retrieval queries
    - LLM: LLM processing (slower) - for summaries, analysis, conversations
    - PATTERN_ANALYSIS: Statistical pattern / trend analysis (Feature 4)
    """
    
    # Keywords that indicate direct lookup queries
    DIRECT_LOOKUP_KEYWORDS = [
        'show', 'list', 'find', 'get', 'display', 'give me', 'what is', 'what are',
        'which', 'search', 'fetch', 'retrieve', 'all', 'filter', 'by'
    ]
    
    # Fields that can be queried directly
    DIRECT_QUERYABLE_FIELDS = [
        'root cause', 'resolution', 'test case', 'test set', 'tcode', 't-code',
        'assignee', 'reporter', 'status', 'severity', 'priority', 'release',
        'opco', 'mtp gate', 'description', 'title', 'summary', 'created',
        'updated', 'attachment', 'comment', 'impacted'
    ]
    
    # Keywords that indicate LLM should be used (full phrases to avoid false positives)
    LLM_KEYWORDS = [
        'summarize', 'summary', 'summarise', 'tell me about', 'explain',
        'analyze', 'analyse', 'describe', 'what happened', 
        'suggest', 'recommend', 'compare', 'difference between'
    ]
    
    # LLM trigger words that need context
    LLM_CONTEXT_WORDS = ['why', 'how']
    
    # ── NEW: Pattern analysis keywords (Feature 4) ──
    PATTERN_ANALYSIS_KEYWORDS = [
        'pattern', 'trend', 'recurring', 'statistics', 'stats', 'insight',
        'breakdown', 'distribution', 'overview', 'analysis', 'analyze',
        'analyse', 'frequent', 'common defect', 'most common',
        'root cause analysis', 'defect trend', 'defect pattern',
        'how many defects', 'defect count', 'workload', 'monthly',
        'timeline', 'time series', 'over time',
    ]

    # Conversational patterns
    CONVERSATIONAL_PATTERNS = [
        r'\b(hi|hello|hey|good morning|good evening|good afternoon)\b',
        r'\bhow are you\b',
        r'\bthanks?\b',
        r'\bthank you\b',
        r'\bhelp\b',
        r'\bbye\b',
        r'\bgoodbye\b'
    ]
    
    def __init__(self):
        pass
    
    def classify(self, query: str) -> Dict[str, Any]:
        """
        Classify a query and return routing decision.
        
        Returns:
            {
                'type': 'DIRECT' | 'LLM' | 'CONVERSATIONAL',
                'confidence': float (0-1),
                'reason': str (explanation of classification),
                'extracted_info': dict (any extracted filters, fields, etc.)
            }
        """
        q_lower = query.lower().strip()
        
        # Check for conversational patterns
        if self._is_conversational(q_lower):
            return {
                'type': 'CONVERSATIONAL',
                'confidence': 1.0,
                'reason': 'Conversational query',
                'extracted_info': {}
            }
        
        # Check for defect creation/template queries
        if self._is_defect_creation_query(q_lower):
            return {
                'type': 'DIRECT',
                'confidence': 1.0,
                'reason': 'Defect creation query',
                'extracted_info': {'action': 'create_defect'}
            }

        # ── NEW: Check for pattern analysis queries (Feature 4) ──
        if self._is_pattern_analysis_query(q_lower):
            return {
                'type': 'PATTERN_ANALYSIS',
                'confidence': 0.95,
                'reason': 'Pattern/trend analysis query',
                'extracted_info': {}
            }
        
        # Extract any filters/criteria
        extracted_info = self._extract_query_info(query)
        
        # Check for direct lookup patterns
        has_direct_keywords = any(kw in q_lower for kw in self.DIRECT_LOOKUP_KEYWORDS)
        has_queryable_fields = any(field in q_lower for field in self.DIRECT_QUERYABLE_FIELDS)
        
        # Check for explicit LLM-requiring keywords (but not if direct lookup is clear)
        has_llm_keywords = any(kw in q_lower for kw in self.LLM_KEYWORDS)
        
        # Prioritize direct lookup if we have clear direct indicators
        if has_direct_keywords and has_queryable_fields:
            return {
                'type': 'DIRECT',
                'confidence': 0.9,
                'reason': 'Query matches direct lookup pattern',
                'extracted_info': extracted_info
            }
        
        # Check for LLM keywords ONLY if not a clear direct query
        if has_llm_keywords and not (has_direct_keywords or has_queryable_fields):
            return {
                'type': 'LLM',
                'confidence': 0.9,
                'reason': 'Query contains LLM keywords (summarize, explain, etc.)',
                'extracted_info': extracted_info
            }
        
        # Check if query is asking for specific defect field
        if self._is_field_specific_query(q_lower):
            return {
                'type': 'DIRECT',
                'confidence': 0.85,
                'reason': 'Query asking for specific defect field',
                'extracted_info': extracted_info
            }
        
        # Check for defect ID in query
        if self._has_defect_id(query):
            if any(kw in q_lower for kw in ['about', 'tell me', 'summary']):
                return {
                    'type': 'LLM',
                    'confidence': 0.8,
                    'reason': 'Query with defect ID asking for summary/explanation',
                    'extracted_info': extracted_info
                }
            else:
                return {
                    'type': 'DIRECT',
                    'confidence': 0.85,
                    'reason': 'Query with defect ID asking for specific information',
                    'extracted_info': extracted_info
                }
        
        # Default to DIRECT for data retrieval, LLM for vague queries
        if has_direct_keywords or has_queryable_fields:
            return {
                'type': 'DIRECT',
                'confidence': 0.7,
                'reason': 'Query appears to be data retrieval',
                'extracted_info': extracted_info
            }
        else:
            return {
                'type': 'LLM',
                'confidence': 0.6,
                'reason': 'Query is vague or complex, using LLM',
                'extracted_info': extracted_info
            }
    
    def _is_conversational(self, query: str) -> bool:
        """Check if query is conversational."""
        for pattern in self.CONVERSATIONAL_PATTERNS:
            if re.search(pattern, query, re.IGNORECASE):
                return True
        return False
    
    def _is_defect_creation_query(self, query: str) -> bool:
        """Check if query is about creating a defect."""
        pattern = r'\b(raise|create|log|new)\b.*\b(defect|bug|issue)\b|\btemplate\b'
        return bool(re.search(pattern, query, re.IGNORECASE))
    
    def _is_field_specific_query(self, query: str) -> bool:
        """Check if query is asking for a specific field."""
        patterns = [
            r'what is (the )?(root cause|resolution|status|tcode)',
            r'(root cause|resolution) (of|for)',
            r'(test cases?|test sets?) (for|of|linked to|impacted by)',
            r'who is (the )?(assignee|reporter)',
            r'when (was|is)',
        ]
        for pattern in patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return True
        return False
    
    # ── NEW (Feature 4) ──
    def _is_pattern_analysis_query(self, query: str) -> bool:
        """Check if query asks for defect pattern analysis / statistics."""
        # Must match at least one pattern keyword and NOT have a specific defect ID
        has_pattern_kw = any(kw in query for kw in self.PATTERN_ANALYSIS_KEYWORDS)
        has_defect_id = bool(re.search(r'\b(COTEST-\d+|SMAR-\d+|D\d{3})\b', query, re.IGNORECASE))
        return has_pattern_kw and not has_defect_id

    def _has_defect_id(self, query: str) -> bool:
        """Check if query contains a defect ID."""
        return bool(re.search(r'\b(COTEST-\d+|SMAR-\d+|D\d{3})\b', query, re.IGNORECASE))
    
    def _extract_query_info(self, query: str) -> Dict[str, Any]:
        """Extract useful information from query."""
        info = {}
        q_lower = query.lower()
        
        # Extract defect IDs
        defect_ids = re.findall(r'\b(COTEST-\d+)\b', query, re.IGNORECASE)
        if defect_ids:
            info['defect_ids'] = [did.upper() for did in defect_ids]
        
        # Extract priority (High, Normal, Low)
        if 'high priority' in q_lower or 'priority high' in q_lower or 'p0' in q_lower or 'p1' in q_lower:
            info['priority'] = 'High'
        elif 'normal priority' in q_lower or 'priority normal' in q_lower or 'p2' in q_lower:
            info['priority'] = 'Normal'
        elif 'low priority' in q_lower or 'priority low' in q_lower or 'p3' in q_lower or 'p4' in q_lower:
            info['priority'] = 'Low'
        
        # Extract severity (Major, Moderate, Minor)
        if 'major' in q_lower or 'critical' in q_lower or 'blocker' in q_lower:
            info['severity'] = 'Major'
        elif 'moderate' in q_lower or 'medium' in q_lower:
            info['severity'] = 'Moderate'
        elif 'minor' in q_lower or 'trivial' in q_lower:
            info['severity'] = 'Minor'
        
        # Extract release (R2, R3.1, R4.1, etc.)
        release_match = re.search(r'\b(R\d+(?:\.\d+)?)\b', query, re.IGNORECASE)
        if release_match:
            info['release'] = release_match.group(1)
        
        # Extract TCode
        tcode_match = re.search(r'\b([A-Z]{2}\d{1,3}[A-Z]?)\b', query)
        if tcode_match:
            info['tcode'] = tcode_match.group(1)
        
        # Extract status
        status_keywords = ['open', 'closed', 'resolved', 'new', 'in progress', 'reopened']
        for status in status_keywords:
            if status in q_lower:
                # Map to actual status values
                if status == 'in progress':
                    info['status'] = 'In Progress'
                elif status == 'reopened':
                    info['status'] = 'Reopened'
                else:
                    info['status'] = status.title()
                break
        
        # Extract assignee (more flexible patterns)
        assignee_patterns = [
            r'assigned to (\w+)',
            r'(\w+)\'s? defects?',
            r'defects? (?:for|by|of) (\w+)',
            r'show (?:me )?(\w+)\'s',
            r'list (\w+)\'s',
        ]
        for pattern in assignee_patterns:
            assignee_match = re.search(pattern, query, re.IGNORECASE)
            if assignee_match:
                name = assignee_match.group(1).upper()
                # Check if it's a known assignee name (not a generic word)
                if name in ['SUBHRA', 'ROSHA', 'SNEHA', 'PINKY', 'SAMUEL', 'SHEEBA']:
                    info['assignee'] = name
                    break
        
        # Extract reporter
        reporter_patterns = [
            r'reported by (\w+)',
            r'reporter (\w+)',
        ]
        for pattern in reporter_patterns:
            reporter_match = re.search(pattern, query, re.IGNORECASE)
            if reporter_match:
                name = reporter_match.group(1).upper()
                if name in ['SUBHRA', 'ROSHA', 'SNEHA', 'PINKY', 'SAMUEL', 'SHEEBA']:
                    info['reporter'] = name
                    break
        
        # Extract date range
        # Pattern 1: "created in December 2025" or "in December 2025"
        month_match = re.search(r'(?:created in|in|for|during) (january|february|march|april|may|june|july|august|september|october|november|december) (\d{4})', query, re.IGNORECASE)
        if month_match:
            month_name = month_match.group(1).capitalize()
            year = month_match.group(2)
            # Convert month name to number
            months = {'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
                     'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12}
            if month_name in months:
                info['date_month'] = months[month_name]
                info['date_year'] = int(year)
        
        # Pattern 2: "created after DD/MM/YYYY" or "created before DD/MM/YYYY"
        date_after_match = re.search(r'(?:after|since) (\d{1,2})[\/\-](\d{1,2})[\/\-](\d{4})', query, re.IGNORECASE)
        if date_after_match:
            info['date_after'] = f"{date_after_match.group(1)}/{date_after_match.group(2)}/{date_after_match.group(3)}"
        
        date_before_match = re.search(r'(?:before|until) (\d{1,2})[\/\-](\d{1,2})[\/\-](\d{4})', query, re.IGNORECASE)
        if date_before_match:
            info['date_before'] = f"{date_before_match.group(1)}/{date_before_match.group(2)}/{date_before_match.group(3)}"
        
        # Detect what field is being requested
        if 'root cause' in q_lower:
            info['requested_field'] = 'RootCause'
        elif 'resolution' in q_lower:
            info['requested_field'] = 'Resolution'
        elif 'test case' in q_lower:
            info['requested_field'] = 'ImpactedTestCases'
        elif 'test set' in q_lower:
            info['requested_field'] = 'ImpactedTestSets'
        elif 'tcode' in q_lower or 't-code' in q_lower:
            info['requested_field'] = 'TCode'
        
        return info
