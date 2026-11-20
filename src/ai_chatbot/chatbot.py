# src/ai_chatbot/chatbot.py
# Enhanced with: ChatResponse (Feature 1), Token tracking (Feature 2),
#                Pattern Analysis routing (Feature 4)

import re
from typing import List, Dict, Any
from dataclasses import dataclass, field
import uuid
from src.utils.rag_helper import answer_with_kedb_rag, answer_with_kedb_rag_with_tokens
from src.repository.defect_repository import DefectRepository
from src.llm.llm_interface import LLMInterface, TokenUsage
from src.ai_chatbot.query_classifier import QueryClassifier
from src.ai_chatbot.direct_query_handler import DirectQueryHandler
from src.ai_chatbot.pattern_analyzer import PatternAnalyzer
from src.repository.history_manager import HistoryManager

# continuation trigger words that indicate stacking on previous context
CONTEXT_CONTINUE_TOKENS = [
    "and their test cases", "and their test case", "and their testcases",
    "and their test sets", "and their test set", "and their tests",
    "and their tcs", "and their t c s", "and test cases", "and test sets",
    "test cases", "test sets", "and their testcases"
]


# ── Feature 1 & 2: ChatResponse dataclass ──
@dataclass
class ChatResponse:
    """Wraps a chatbot answer with metadata for the UI."""
    answer: str = ""
    token_usage: TokenUsage = None
    llm_used: bool = False
    follow_ups: List[str] = field(default_factory=list)

    def __post_init__(self):
        if self.token_usage is None:
            self.token_usage = TokenUsage()


class ChatBot:
    def __init__(self, model_choice="llama3", csv_path="db/defects_kedb.csv"):
        self.defect_repo = DefectRepository()
        self.llm = LLMInterface(model_choice=model_choice)
        self.query_classifier = QueryClassifier()
        self.direct_handler = DirectQueryHandler(csv_path=csv_path)
        self.pattern_analyzer = PatternAnalyzer(csv_path=csv_path)   # Feature 4
        self.history_manager = HistoryManager()
        self.session_id = str(uuid.uuid4())
        self.search_mode = "Hybrid"
        self.last_defects: List[Dict[str, Any]] = []
        print("[ChatBot] Initialized with query optimization enabled")

    # ────────────────────────────────────────────────
    # Helper: detect broad query type (legacy, kept)
    # ────────────────────────────────────────────────
    def detect_query_type(self, query: str) -> str:
        q = (query or "").lower()
        if any(w in q for w in ["defect", "bug", "issue", "error", "d0", "d1", "d2", "d3"]):
            return "defect"
        elif any(w in q for w in ["test case", "testcycle", "zephyr", "execution", "test set", "testcases"]):
            return "testcase"
        elif any(w in q for w in ["release", "r3.", "r4.", "build", "version", "deployment"]):
            return "release"
        else:
            return "general"

    # ────────────────────────────────────────────────
    # Gibberish detector (unchanged)
    # ────────────────────────────────────────────────
    def _is_gibberish(self, query: str) -> bool:
        import re, math
        from collections import Counter

        if not query:
            return True
        q = query.strip()
        if len(q) < 3:
            return True
        if re.search(r"\b(COTEST-\d+|SMAR-\d+|D\d{3})\b", q, flags=re.IGNORECASE):
            return False
        if re.search(r"\bt-?code[:\s]*[A-Z0-9_\-]{2,10}\b", q, flags=re.IGNORECASE):
            return False
        non_alnum = sum(1 for c in q if not (c.isalnum() or c.isspace()))
        if non_alnum / max(1, len(q)) > 0.35:
            return True
        letters_only = re.sub(r'[^A-Za-z\s]', ' ', q).strip()
        if not letters_only:
            return True
        tokens = [t for t in re.findall(r"[A-Za-z]+", letters_only) if t]
        if not tokens:
            return True
        vowel_tokens = sum(1 for t in tokens if re.search(r'[aeiou]', t, flags=re.I))
        if len(tokens) >= 2 and (vowel_tokens / len(tokens)) < 0.35:
            return True
        avg_len = sum(len(t) for t in tokens) / len(tokens)
        if avg_len < 2.0:
            return True
        s = ''.join(tokens).lower()
        counts = Counter(s)
        probs = [c / len(s) for c in counts.values()]
        entropy = -sum(p * math.log2(p) for p in probs) if probs else 0.0
        if entropy < 2.3:
            return True
        bigrams = [s[i:i+2] for i in range(len(s)-1)]
        if bigrams:
            unique_ratio = len(set(bigrams)) / len(bigrams)
            if unique_ratio < 0.45:
                return True
        return False

    # ────────────────────────────────────────────────
    # Pattern analysis keywords (Feature 4)
    # ────────────────────────────────────────────────
    PATTERN_KEYWORDS = [
        'pattern', 'trend', 'recurring', 'common', 'analysis', 'analyze', 'analyse', 'distribution',
        'statistics', 'stats', 'overview', 'insight', 'breakdown', 'frequent',
        'most common', 'dashboard', 'by severity', 'by status'
    ]

    def _is_pattern_query(self, query: str) -> bool:
        q = query.lower()
        return any(kw in q for kw in self.PATTERN_KEYWORDS)

    # ────────────────────────────────────────────────
    # Feature 1: Generate follow-up suggestions
    # ────────────────────────────────────────────────
    def _generate_follow_ups(self, query: str, answer: str) -> List[str]:
        """Create 2-3 context-aware follow-up suggestions."""
        q_lower = query.lower()
        suggestions: List[str] = []

        # Extract defect IDs from the answer or query
        defect_ids = re.findall(r'\bCOTEST-\d+\b', answer + " " + query, re.IGNORECASE)
        defect_ids = list(dict.fromkeys([d.upper() for d in defect_ids]))  # unique, ordered

        if defect_ids:
            did = defect_ids[0]
            if 'root cause' not in q_lower:
                suggestions.append(f"What is the root cause of {did}?")
            if 'resolution' not in q_lower:
                suggestions.append(f"Show resolution for {did}")
            if 'test case' not in q_lower and 'test set' not in q_lower:
                suggestions.append(f"What test cases are impacted by {did}?")

        # Pattern / trend suggestions
        if self._is_pattern_query(query):
            if 'trend' not in q_lower:
                suggestions.append("Show defect trends over time")
            if 'severity' not in q_lower:
                suggestions.append("Show severity breakdown")
            if 'root cause' not in q_lower:
                suggestions.append("What are the common root causes?")
        elif not defect_ids:
            # Generic helpful suggestions
            suggestions.append("Show defect pattern overview")
            suggestions.append("List all open defects")
            suggestions.append("Show recurring defects")

        return suggestions[:3]


    def _extract_defect_ids(self, text: str) -> List[str]:
        ids = re.findall(r"\bCOTEST-\d+\b", text or "", re.IGNORECASE)
        return list(dict.fromkeys([x.upper() for x in ids]))

    def _save_interaction_history(self, query: str, response: ChatResponse):
        """Persist each chatbot interaction in conversation history."""
        try:
            defect_ids = self._extract_defect_ids((query or "") + " " + (response.answer or ""))
            tokens = {
                "prompt_tokens": response.token_usage.prompt_tokens if response.token_usage else 0,
                "completion_tokens": response.token_usage.completion_tokens if response.token_usage else 0,
                "total_tokens": response.token_usage.total_tokens if response.token_usage else 0,
                "llm_used": response.llm_used,
            }
            self.history_manager.save_conversation(
                query=query or "",
                response=response.answer or "",
                session_id=self.session_id,
                tokens=tokens,
                defect_ids=defect_ids,
                
            )
        except Exception as e:
            print(f"[ChatBot] Warning: failed to persist conversation history: {e}")

    def _llm_failed(self, text: str) -> bool:
        low = (text or "").lower()
        return (
            "generation failed" in low
            or "failed to connect to ollama" in low
            or low.startswith("⚠️ ollama")
        )

    def _safe_legacy_fallback(self, query: str, default_text: str = "") -> str:
        """Best-effort deterministic fallback when LLM path fails."""
        try:
            legacy = self._legacy_answer(query)
            if legacy and legacy.strip() and not self._llm_failed(legacy):
                return legacy
        except Exception as e:
            print(f"[ChatBot] Legacy fallback error: {e}")

        # Secondary fallback: semantic retrieval without generation.
        try:
            defects = self.defect_repo.query_defects(query, k=5, mode=self.search_mode)
            if defects:
                blocks = [self.defect_repo.format_full_record_markdown(d) for d in defects[:5]]
                return "\n---\n".join(blocks)
        except Exception as e:
            print(f"[ChatBot] Repository fallback error: {e}")

        # Tertiary fallback: deterministic keyword search on CSV text fields.
        try:
            keyword_result = self.direct_handler.keyword_search(query, top_k=5)
            if keyword_result:
                return keyword_result
        except Exception as e:
            print(f"[ChatBot] Keyword fallback error: {e}")

        if default_text and self._llm_failed(default_text):
            return "⚠️ I couldn't find enough matching defect context to answer this reliably. Try adding a defect ID, TCode, release, or status filter."

        return default_text or "⚠️ I could not generate an LLM response. Please refine the query."

    def _build_single_defect_summary(self, defect_id: str, defect: Dict[str, Any]) -> str:
        """Deterministic summary used when LLM is unavailable."""
        return (
            f"**Summary for {defect_id}:**\n\n"
            f"- **What happened:** {defect.get('Summary', 'Not specified')}\n"
            f"- **Severity/Impact:** Severity {defect.get('Severity', 'N/A')}, Priority {defect.get('Priority', 'N/A')}\n"
            f"- **Probable Root Cause:** {defect.get('RootCause', 'Not specified')}\n"
            f"- **Resolution Status:** {defect.get('Resolution', 'Not specified')}\n"
            f"- **Next steps:** Verify impacted tests ({defect.get('ImpactedTestCases', 'N/A')}) and closure readiness."
        )

    def _build_multi_defect_comparison(self, defect_ids: List[str]) -> str:
        """Structured comparison for multi-defect requests without LLM dependency."""
        unique_ids = list(dict.fromkeys([d.upper() for d in defect_ids]))
        defects = []
        for did in unique_ids:
            d = self.direct_handler.get_defect_details(did)
            if d:
                defects.append((did, d))

        if not defects:
            return f"❌ None of the requested defects were found: {', '.join(unique_ids)}"

        lines = ["📊 **Defect Comparison**\n"]
        for did, d in defects:
            lines.append(
                f"**{did}** | Status: {d.get('Status', 'N/A')} | Severity: {d.get('Severity', 'N/A')} | "
                f"TCode: {d.get('TCode', 'N/A')} | Release: {d.get('Release', 'N/A')}"
            )
            lines.append(f"- Root Cause: {d.get('RootCause', 'Not specified')}")
            lines.append(f"- Resolution: {d.get('Resolution', 'Not specified')}")
            lines.append(f"- Impacted Test Cases: {d.get('ImpactedTestCases', 'Not specified')}\n")

        if len(defects) >= 2:
            root_causes = {d.get('RootCause', '').strip().lower() for _, d in defects if d.get('RootCause')}
            tcodes = {d.get('TCode', '').strip().upper() for _, d in defects if d.get('TCode') and d.get('TCode') != 'N/A'}
            if len(root_causes) == 1 and root_causes:
                lines.append("**Observation:** These defects appear to share a similar root cause.")
            if len(tcodes) == 1 and tcodes:
                lines.append(f"**Observation:** All compared defects are tied to transaction `{next(iter(tcodes))}`.")

        missing = [did for did in unique_ids if did not in {x[0] for x in defects}]
        if missing:
            lines.append(f"\n⚠️ Not found: {', '.join(missing)}")

        return "\n".join(lines).strip()
    def _is_legacy_rich_query(self, query: str) -> bool:
        """Queries that were heavily handled in legacy chatbot logic."""
        q = (query or "").lower()
        legacy_markers = [
            "tcode", "t-code", "test case", "test cases", "test set", "test sets",
            "root cause", "resolution", "opco", "severity", "status", "release",
            "impacted", "show details", "full details", "list p1", "list p2", "list p3", "list p4"
        ]
        return any(m in q for m in legacy_markers)

    def _legacy_answer(self, query: str) -> str:
        """Preserves rich legacy retrieval behavior from earlier chatbot versions."""
        q_raw = query or ""
        q_lower = q_raw.lower().strip()

        query_type = self.detect_query_type(query)
        if query_type == "defect":
            self.search_mode = "Hybrid"
        elif query_type == "testcase":
            self.search_mode = "Semantic Search"
        elif query_type == "release":
            return answer_with_kedb_rag(query)
        else:
            self.search_mode = "Hybrid"

        FOLLOWUP_KEYWORDS = [
            "test case", "test cases", "test set", "test sets",
            "impacted", "resolution", "root cause", "what is the resolution",
            "what's the resolution", "give the resolution", "their resolution",
            "that defect", "the defect", "and their test cases", "and their test sets",
            "tcode", "t-code", "t code", "release"
        ]

        has_defect_id = bool(self.defect_repo.extract_single_defect_id(query))
        is_followup = (not has_defect_id) and any(token in q_lower for token in FOLLOWUP_KEYWORDS)

        if is_followup and self.last_defects:
            defects = self.last_defects
        else:
            defect_id = self.defect_repo.extract_single_defect_id(query)
            if defect_id:
                defects = self.defect_repo.query_defects(defect_id, k=10, mode=self.search_mode)
            else:
                defects = self.defect_repo.query_defects(query, k=10, mode=self.search_mode)
            if defects:
                self.last_defects = defects

        # Enhanced TCode query handling
        if "tcode" in q_lower or "t-code" in q_lower:
            tcode_match = re.search(r"t-?code[:\s]*([A-Z0-9_\-]{2,10})", query, flags=re.IGNORECASE)
            tcode_token = tcode_match.group(1).upper() if tcode_match else None

            sev_map = {"p1": "P1", "p2": "P2", "p3": "P3", "p4": "P4"}
            opcos = {"dll", "ds", "mi", "ab"}
            statuses = {"open", "closed", "resolved", "in progress"}

            where_filter = {}
            if tcode_token:
                where_filter["TCode"] = tcode_token

            for sk, sval in sev_map.items():
                if re.search(rf"\b{sk}\b", q_lower):
                    where_filter["Severity"] = sval
                    break
            for oc in opcos:
                if re.search(rf"\b{oc}\b", q_lower):
                    where_filter["OpCo"] = oc.upper()
                    break
            for st in statuses:
                if re.search(rf"\b{st}\b", q_lower):
                    where_filter["Status"] = st.title() if st != "in progress" else "In Progress"
                    break

            release_match = re.search(r"\b(R\d+(?:\.\d+){1,2})\b", query, flags=re.IGNORECASE)
            if release_match:
                where_filter["Release"] = release_match.group(1).upper()

            if where_filter:
                filtered_defects = self.defect_repo.query_defects(query, k=10, mode=self.search_mode, where=where_filter)
                if filtered_defects:
                    details = [self.defect_repo.format_full_record_markdown(d) for d in filtered_defects]
                    return f"**Defects impacted by TCode {tcode_token or ''}:**\n\n" + "\n---\n".join(details)

        # Requested fields/listing logic from legacy behavior
        requested_fields = self.defect_repo.extract_requested_fields(query)
        list_intent = self.defect_repo.is_list_intent(query)
        wants = self.defect_repo.wants_testcases_or_testsets(query)

        if requested_fields == ["all"] or re.search(r"\b(details|full details|give details|show details|about)\b", q_lower):
            details = [self.defect_repo.format_full_record_markdown(d) for d in defects]
            return "\n---\n".join(details)

        if requested_fields and requested_fields != ["all"]:
            if wants["testcases"] and "ImpactedTestCases" not in requested_fields:
                requested_fields.append("ImpactedTestCases")
            formatted_blocks = [self.defect_repo.format_selected_fields_markdown(d, requested_fields) for d in defects]
            return "\n---\n".join(formatted_blocks)

        if list_intent:
            ids_unique = []
            seen = set()
            for d in defects:
                did = d.get("DefectID") or d.get("IssueKey")
                if did and did not in seen:
                    seen.add(did)
                    ids_unique.append(did)
            return "\n".join(ids_unique) if ids_unique else "⚠️ No matching DefectIDs found."

        if wants["testsets"]:
            blocks = []
            seen_ids = set()
            for d in defects:
                did = d.get("DefectID") or d.get("IssueKey")
                if did in seen_ids:
                    continue
                seen_ids.add(did)
                header = f"**DefectID:** {did}  \n"
                blocks.append(header + self.defect_repo.format_testsets_block(d))
            return "\n---\n".join(blocks)

        if wants["testcases"]:
            return self.defect_repo.format_testcases_block(defects)

        if not defects:
            return answer_with_kedb_rag(query) or "⚠️ No matching records found."

        # default id list
        ids_unique = []
        seen = set()
        for d in defects:
            did = d.get("DefectID") or d.get("IssueKey")
            if did and did not in seen:
                seen.add(did)
                ids_unique.append(did)
        return "\n".join(ids_unique) if ids_unique else "⚠️ No matching DefectIDs found."


    # ────────────────────────────────────────────────
    # ask() wrapper (kept for backward compatibility)
    # ────────────────────────────────────────────────
    def ask(self, query: str):
        """Backward-compatible API returning answer text only."""
        return self.ask_with_tokens(query).answer

    # ────────────────────────────────────────────────
    # Feature 1 + 2: ask_with_tokens() – returns ChatResponse
    # ────────────────────────────────────────────────
    def ask_with_tokens(self, query: str) -> 'ChatResponse':
        """
        Same routing logic as ask(), but returns a ChatResponse with
        token usage and follow-up suggestions.
        """
        q_raw = query or ""
        q_lower = q_raw.lower().strip()

        print(f"[ChatBot] Processing query (with tokens): {query[:50]}...")

        # ---- Small talk & help ----
        greetings = ["hi", "hello", "hey", "good morning", "good evening", "good afternoon"]
        greet_pattern = re.compile(r'\b(' + '|'.join(re.escape(g) for g in greetings) + r')\b', flags=re.IGNORECASE)
        how_are_you_pattern = re.compile(r'\b(how are you|how r u)\b', flags=re.IGNORECASE)
        HELP_PATTERN = re.compile(r'\bhelp\b', flags=re.IGNORECASE)
        RAISE_DEFECT_PATTERN = re.compile(
            r'\b(raise|create|log)\b.*\b(defect|bug|issue)\b|\bnew defect\b|\btemplate\b',
            flags=re.IGNORECASE
        )

        def _quick_response(text: str) -> ChatResponse:
            follow = self._generate_follow_ups(query, text)
            return ChatResponse(answer=text, llm_used=False, follow_ups=follow)

        def _finalize(resp: ChatResponse) -> ChatResponse:
            self._save_interaction_history(query, resp)
            return resp

        if RAISE_DEFECT_PATTERN.search(q_lower):
            return _finalize(_quick_response(self.generate_defect_template(None)))

        if greet_pattern.search(q_lower):
            if how_are_you_pattern.search(q_lower):
                return _finalize(_quick_response("I'm just code running on a server, but I'm working great! 😊 How can I help you with defects, releases, or test cases today?"))
            return _finalize(_quick_response("👋 Hello! How can I assist you with defects, releases, or test cases today?"))

        if how_are_you_pattern.search(q_lower):
            return _finalize(_quick_response("I'm doing well, thanks for asking! How can I help you with defects, releases, or test cases?"))

        if "thank" in q_lower:
            return _finalize(_quick_response("🙏 You're welcome!"))

        if HELP_PATTERN.search(q_lower):
            return _finalize(_quick_response(
                "💡 You can ask things like:\n"
                "- 'Show me defects related to invoice duplication'\n"
                "- 'Give me details of defect COTEST-0001'\n"
                "- 'List P1 defects from DLL OpCo'\n"
                "- 'What is the root cause of COTEST-0002?'\n"
                "- 'Show resolution for COTEST-0003'\n"
                "- 'List defects assigned to SAMUEL'\n"
                "- 'Show defects for TCode ME23N'\n"
                "- 'Test cases linked to COTEST-0002'\n"
                "- 'Show release-wise defect summary'\n"
                "- 'Show defect pattern overview'\n"
                "- 'Raise a new defect' or 'Create defect'"
            ))

        if self._is_gibberish(q_raw):
            return _finalize(_quick_response("❌ Sorry — I couldn't understand your message. Please ask about defects or provide a clearer query."))

        # Legacy-rich query fallback to preserve pre-existing capabilities
        if self._is_legacy_rich_query(query):
            try:
                legacy_answer = self._legacy_answer(query)
                if legacy_answer:
                    follow = self._generate_follow_ups(query, legacy_answer)
                    return _finalize(ChatResponse(answer=legacy_answer, llm_used=False, follow_ups=follow))
            except Exception as e:
                print(f"[ChatBot] Legacy fallback failed: {e}")

        # ---- Classify ----
        classification = self.query_classifier.classify(query)
        query_type = classification['type']
        extracted_info = classification['extracted_info']

        print(f"[ChatBot] Query classified as: {query_type} (confidence: {classification['confidence']:.2f})")

        # Explicit deterministic compare path for multi-defect comparison queries.
        defect_ids = extracted_info.get('defect_ids', [])
        if len(defect_ids) >= 2 and any(k in q_lower for k in ['compare', 'difference', 'differences']):
            text = self._build_multi_defect_comparison(defect_ids)
            follow = self._generate_follow_ups(query, text)
            return _finalize(ChatResponse(answer=text, llm_used=False, follow_ups=follow))

        # ── Feature 4: Pattern analysis ──
        if query_type == 'PATTERN_ANALYSIS' or (self._is_pattern_query(query) and 'defect_ids' not in extracted_info):
            answer = self.pattern_analyzer.analyze(query)
            follow = self._generate_follow_ups(query, answer)
            return _finalize(ChatResponse(answer=answer, llm_used=False, follow_ups=follow))

        if query_type == 'CONVERSATIONAL':
            return _finalize(_quick_response("👋 Hello! How can I assist you with defects, releases, or test cases today?"))

        elif query_type == 'DIRECT':
            print("[ChatBot] Using DIRECT query handler (fast)")

            if extracted_info.get('action') == 'create_defect':
                return _finalize(_quick_response(self.generate_defect_template(None)))

            if 'defect_ids' in extracted_info and len(extracted_info['defect_ids']) == 1:
                defect_id = extracted_info['defect_ids'][0]
                defect = self.direct_handler.get_defect_details(defect_id)
                if defect:
                    if 'requested_field' in extracted_info:
                        field = extracted_info['requested_field']
                        field_value = defect.get(field, 'Not specified')
                        text = f"**{defect_id}** - {defect['Summary'][:60]}\n\n**{field}:** {field_value}"
                    else:
                        text = self.direct_handler.format_defect_details(defect)
                    follow = self._generate_follow_ups(query, text)
                    return _finalize(ChatResponse(answer=text, llm_used=False, follow_ups=follow))
                else:
                    return _finalize(_quick_response(f"❌ Defect {defect_id} not found in the database."))

            try:
                result = self.direct_handler.query(extracted_info)
                if isinstance(result, str) and result.startswith("❌ No defects found"):
                    original_no_result = result
                    try:
                        legacy_result = self._legacy_answer(query)
                        result = legacy_result if (legacy_result and legacy_result.strip()) else original_no_result
                    except Exception:
                        result = original_no_result
                follow = self._generate_follow_ups(query, result)
                return _finalize(ChatResponse(answer=result, llm_used=False, follow_ups=follow))
            except Exception as e:
                print(f"[ChatBot] Direct query error: {e}, falling back to LLM")
                text, usage = answer_with_kedb_rag_with_tokens(query)
                follow = self._generate_follow_ups(query, text)
                return _finalize(ChatResponse(answer=text, token_usage=usage, llm_used=True, follow_ups=follow))

        else:  # LLM
            print("[ChatBot] Using LLM for complex query (slower)")

            query_type_internal = self.detect_query_type(query)
            if query_type_internal == "release":
                text, usage = answer_with_kedb_rag_with_tokens(query)
                if self._llm_failed(text):
                    text = self._safe_legacy_fallback(query, default_text=text)
                    follow = self._generate_follow_ups(query, text)
                    return _finalize(ChatResponse(answer=text, llm_used=False, follow_ups=follow))
                follow = self._generate_follow_ups(query, text)
                return _finalize(ChatResponse(answer=text, token_usage=usage, llm_used=True, follow_ups=follow))

            if 'defect_ids' in extracted_info and len(extracted_info['defect_ids']) == 1:
                defect_id = extracted_info['defect_ids'][0]
                defect = self.direct_handler.get_defect_details(defect_id)
                if defect:
                    prompt = (
                        "You are a concise QA assistant. Summarize the defect below in 5 bullet points: "
                        "(what happened, severity/impact, probable root cause, resolution status, next steps)\n\n"
                        f"{self.direct_handler.format_defect_details(defect)}\n\nSummary:"
                    )
                    try:
                        llm_text, usage = self.llm.generate_with_tokens(prompt)
                        if self._llm_failed(llm_text):
                            fallback = self._build_single_defect_summary(defect_id, defect)
                            follow = self._generate_follow_ups(query, fallback)
                            return _finalize(ChatResponse(answer=fallback, llm_used=False, follow_ups=follow))

                        answer = f"**Summary for {defect_id}:**\n\n{llm_text}"
                        follow = self._generate_follow_ups(query, answer)
                        return _finalize(ChatResponse(answer=answer, token_usage=usage, llm_used=True, follow_ups=follow))
                    except Exception as e:
                        print(f"[ChatBot] LLM error: {e}")
                        fallback = self._build_single_defect_summary(defect_id, defect)
                        follow = self._generate_follow_ups(query, fallback)
                        return _finalize(ChatResponse(answer=fallback, llm_used=False, follow_ups=follow))
                else:
                    return _finalize(_quick_response(f"❌ Defect {defect_id} not found in the database."))

            text, usage = answer_with_kedb_rag_with_tokens(query)
            if self._llm_failed(text):
                text = self._safe_legacy_fallback(query, default_text=text)
                follow = self._generate_follow_ups(query, text)
                return _finalize(ChatResponse(answer=text, llm_used=False, follow_ups=follow))

            follow = self._generate_follow_ups(query, text)
            return _finalize(ChatResponse(answer=text, token_usage=usage, llm_used=True, follow_ups=follow))

    # ────────────────────────────────────────────────
    # Session token helpers (Feature 2)
    # ────────────────────────────────────────────────
    def get_session_token_stats(self) -> dict:
        return self.llm.get_session_stats()

    def reset_session_stats(self):
        self.llm.reset_session_stats()

    # ────────────────────────────────────────────────
    # Defect template (unchanged)
    # ────────────────────────────────────────────────
    def generate_defect_template(self, defect_id=None):
        id_line = f"Defect ID: {defect_id}" if defect_id else "Defect ID: [New - assign ID]"
        return (
            f"{id_line}\n"
            "Please note, you can use the <b>'Create Defect'</b> option on the left to raise a new defect. Fill in the required details like title, description, and severity\n\n"
            "<b>Summary:</b> [Short one-line description of the issue]\n"
            "<b>Description:</b> [Detailed description of the issue]\n"
            "<b>Environment:</b>\n"
            "<b>Application / Module</b>: [app/module]\n"
            "<b>Test data</b>: [test data]\n\n"
            "<b>Steps to Reproduce</b>:\n"
            "1. [Step 1 with exact input]\n"
            "2. [Step 2]\n"
            "...\n"
            "<b>Expected result</b>: [what should happen]\n"
            "<b>Actual result</b>: [what happens - include screenshots/logs if available]\n\n"
            "<b>Severity</b> : [Severity (Blocker/Critical/Major/Minor/Trivial) / Priority (P0/P1/P2/P3)]\n\n"
            "<b>Screenshots / Attachments</b>:\n"
            "<b>Regression</b>:\n"
            "<b>New issue or regression? [Yes/No]</b>\n"
            "<b>If regression, last known good build/version: [version]</b>\n\n"
            "<b>Assigned To</b>\n"
            "<b>Reported By</b>\n"
        )