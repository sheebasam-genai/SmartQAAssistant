# src/repository/history_manager.py

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4


class HistoryManager:
    """Persist and query conversation + defect creation history."""

    def __init__(self, history_path: str = "db/conversation_history.json"):
        self.history_path = Path(history_path)
        self._ensure_history_file()

    def _ensure_history_file(self):
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.history_path.exists():
            self._write_data({"conversations": [], "defect_history": []})

    def _read_data(self) -> Dict[str, Any]:
        try:
            with open(self.history_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            data = {"conversations": [], "defect_history": []}

        if "conversations" not in data:
            data["conversations"] = []
        if "defect_history" not in data:
            data["defect_history"] = []
        return data

    def _write_data(self, data: Dict[str, Any]):
        with open(self.history_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def save_conversation(
        self,
        query: str,
        response: str,
        session_id: str,
        tokens: Optional[Dict[str, Any]] = None,
        defect_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Save a single chat interaction."""
        data = self._read_data()

        item = {
            "id": str(uuid4()),
            "timestamp": datetime.utcnow().isoformat(),
            "user_query": query or "",
            "assistant_response": response or "",
            "tokens_used": tokens or {},
            "defect_ids_mentioned": defect_ids or [],
        }

        session = next(
            (c for c in data["conversations"] if c.get("session_id") == session_id),
            None
        )

        message = item

        if session:
            session["messages"].append(message)
        else:
            data["conversations"].append({
                "session_id": session_id,
                "messages": [message]
            })
        self._write_data(data)
        return item

    def save_defect_creation(self, defect_data: Dict[str, Any]) -> Dict[str, Any]:
        """Save created defect metadata."""
        data = self._read_data()

        item = {
            "defect_id": defect_data.get("defect_id", ""),
            "created_at": defect_data.get("created_at", datetime.utcnow().isoformat()),
            "title": defect_data.get("title", ""),
            "description": defect_data.get("description", ""),
            "created_by": defect_data.get("created_by", ""),
            "ai_generated_title": bool(defect_data.get("ai_generated_title", False)),
        }

        data["defect_history"].append(item)
        self._write_data(data)
        return item

    def search_history(
        self,
        keyword: Optional[str] = None,
        date_range: Optional[Dict[str, str]] = None,
        defect_id: Optional[str] = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Search across conversation and defect history.
        date_range format: {"start": "ISO", "end": "ISO"}
        """
        data = self._read_data()
        kw = (keyword or "").lower().strip()
        defect_id_upper = (defect_id or "").upper().strip()

        start_dt = None
        end_dt = None
        if date_range:
            try:
                if date_range.get("start"):
                    start_dt = datetime.fromisoformat(date_range["start"])
                if date_range.get("end"):
                    end_dt = datetime.fromisoformat(date_range["end"])
            except Exception:
                start_dt, end_dt = None, None

        def in_date_range(ts: str) -> bool:
            if not (start_dt or end_dt):
                return True
            try:
                ts_dt = datetime.fromisoformat(ts)
            except Exception:
                return False
            if start_dt and ts_dt < start_dt:
                return False
            if end_dt and ts_dt > end_dt:
                return False
            return True

        filtered_conversations: List[Dict[str, Any]] = []
        for item in data["conversations"]:
            if not in_date_range(item.get("timestamp", "")):
                continue

            blob = " ".join([
                item.get("user_query", ""),
                item.get("assistant_response", ""),
                " ".join(item.get("defect_ids_mentioned", [])),
            ]).lower()

            if kw and kw not in blob:
                continue

            if defect_id_upper:
                ids = [x.upper() for x in item.get("defect_ids_mentioned", [])]
                if defect_id_upper not in ids and defect_id_upper.lower() not in blob:
                    continue

            filtered_conversations.append(item)

        filtered_defects: List[Dict[str, Any]] = []
        for item in data["defect_history"]:
            if not in_date_range(item.get("created_at", "")):
                continue

            blob = " ".join([
                item.get("defect_id", ""),
                item.get("title", ""),
                item.get("description", ""),
                item.get("created_by", ""),
            ]).lower()

            if kw and kw not in blob:
                continue

            if defect_id_upper and item.get("defect_id", "").upper() != defect_id_upper:
                continue

            filtered_defects.append(item)

        return {
            "conversations": filtered_conversations,
            "defect_history": filtered_defects,
        }

    def get_recent_conversations(self, limit: int = 10) -> List[Dict[str, Any]]:
        data = self._read_data()
        conv = data.get("conversations", [])
        return list(reversed(conv[-max(limit, 0):]))
