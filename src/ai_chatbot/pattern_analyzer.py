# src/ai_chatbot/pattern_analyzer.py
# ── NEW: Defect Pattern Analysis module (Feature 4)
# Provides statistical insights, trend analysis, recurring defect detection,
# and common root cause identification from the defects_kedb.csv data.

import pandas as pd
import re
from typing import Dict, Any, List, Optional
from pathlib import Path
from collections import Counter
from datetime import datetime


class PatternAnalyzer:
    """
    Analyses defect patterns from the CSV knowledge base.
    Capabilities:
      - Recurring defect identification
      - Trend analysis over time
      - Common root cause analysis
      - Statistical insights (by severity, priority, status, assignee, release)
    """

    def __init__(self, csv_path: str = "db/defects_kedb.csv"):
        self.csv_path = Path(csv_path)
        self.df = None
        self._load_data()

    def _load_data(self):
        """Load and clean CSV data."""
        try:
            if self.csv_path.exists():
                self.df = pd.read_csv(self.csv_path, keep_default_na=False)
                self.df.columns = self.df.columns.str.strip()
                # Parse dates
                self.df['_created_dt'] = pd.to_datetime(
                    self.df['Created'], format='%d/%m/%Y %H:%M', errors='coerce'
                )
                print(f"[PatternAnalyzer] Loaded {len(self.df)} rows")
            else:
                self.df = pd.DataFrame()
        except Exception as e:
            print(f"[PatternAnalyzer] Error loading CSV: {e}")
            self.df = pd.DataFrame()

    def reload_data(self):
        self._load_data()

    # ──────────────────────────────────────────
    # Query dispatcher
    # ──────────────────────────────────────────
    def analyze(self, query: str) -> str:
        """
        Main entry point – route the user's pattern/analysis question
        to the right sub-analysis and return a formatted answer.
        """
        q = query.lower()

        if self.df is None or self.df.empty:
            return "❌ No defect data available for pattern analysis."

        # Determine which kind of analysis the user wants
        if any(w in q for w in ['recurring', 'repeat', 'frequent', 'common defect', 'most common']):
            return self._recurring_defects()
        elif any(w in q for w in ['trend', 'over time', 'monthly', 'timeline', 'time series']):
            return self._trend_analysis()
        elif any(w in q for w in ['root cause', 'common cause', 'why', 'causes']):
            return self._root_cause_analysis()
        elif any(w in q for w in ['severity', 'critical', 'major', 'moderate', 'minor']):
            return self._severity_breakdown()
        elif any(w in q for w in ['assignee', 'assigned', 'workload', 'who']):
            return self._assignee_analysis()
        elif any(w in q for w in ['release', 'gate', 'version']):
            return self._release_analysis()
        elif any(w in q for w in ['status', 'open', 'closed', 'new', 'resolved']):
            return self._status_breakdown()
        elif any(w in q for w in ['tcode', 'transaction', 't-code']):
            return self._tcode_analysis()
        elif any(w in q for w in ['overview', 'summary', 'stats', 'statistics', 'insight', 'dashboard', 'pattern']):
            return self._full_overview()
        else:
            # Default: provide full overview
            return self._full_overview()

    # ──────────────────────────────────────────
    # Individual analysis methods
    # ──────────────────────────────────────────

    def _recurring_defects(self) -> str:
        """Identify defects that share similar summaries (recurring issues)."""
        df = self.df[self.df['Summary'].str.strip() != ''].copy()
        if df.empty:
            return "No defect summaries available for recurring analysis."

        # Extract keywords from summaries and group
        keyword_counter: Counter = Counter()
        tcode_counter: Counter = Counter()

        for _, row in df.iterrows():
            summary = str(row.get('Summary', ''))
            # Extract tcodes
            tcodes = re.findall(r'\b([A-Z]{2,4}\d{1,3}[A-Z]?)\b', summary)
            for tc in tcodes:
                tcode_counter[tc] += 1
            # Extract meaningful words (>3 chars)
            words = re.findall(r'\b[a-zA-Z]{4,}\b', summary.lower())
            stopwords = {'during', 'automation', 'execution', 'transaction', 'unexpected',
                         'issue', 'occurred', 'system', 'halted', 'unexpectedly', 'with', 'when',
                         'screen', 'returned', 'blank', 'that', 'this', 'from', 'into', 'test'}
            for w in words:
                if w not in stopwords:
                    keyword_counter[w] += 1

        lines = ["📊 **Recurring Defect Patterns**\n"]

        # Most common transaction codes with issues
        if tcode_counter:
            lines.append("**Most Affected Transaction Codes:**")
            for tc, count in tcode_counter.most_common(10):
                lines.append(f"  • `{tc}` – {count} defect(s)")
            lines.append("")

        # Most common keywords
        if keyword_counter:
            lines.append("**Frequently Mentioned Terms:**")
            for kw, count in keyword_counter.most_common(10):
                if count >= 2:
                    lines.append(f"  • *{kw}* – appears in {count} defect(s)")
            lines.append("")

        # Defects with same status "Open" or "Reopened" (unresolved)
        unresolved = df[df['Status'].str.lower().isin(['open', 'new', 'reopened', 'in progress'])]
        if not unresolved.empty:
            lines.append(f"**Unresolved Defects:** {len(unresolved)} out of {len(df)} total")

        return "\n".join(lines)

    def _trend_analysis(self) -> str:
        """Analyse defect trends over time (monthly)."""
        df = self.df.dropna(subset=['_created_dt']).copy()
        if df.empty:
            return "No date information available for trend analysis."

        df['_month'] = df['_created_dt'].dt.to_period('M')
        monthly = df.groupby('_month').size().sort_index()

        lines = ["📈 **Defect Trend (Monthly)**\n"]
        for period, count in monthly.items():
            bar = "█" * min(count, 30)
            lines.append(f"  {period}  {bar} ({count})")

        # Peak month
        if not monthly.empty:
            peak = monthly.idxmax()
            lines.append(f"\n**Peak month:** {peak} with {monthly.max()} defect(s)")

        return "\n".join(lines)

    def _root_cause_analysis(self) -> str:
        """Extract and count common root causes from Comments."""
        causes: List[str] = []
        for _, row in self.df.iterrows():
            comments = str(row.get('Comments', ''))
            # Look for "Root Cause:" pattern
            match = re.search(r'Root\s*Cause[:\s]+(.+?)(?:\.|;|\n|$)', comments, re.IGNORECASE)
            if match:
                cause = match.group(1).strip()
                if cause and len(cause) > 5:
                    causes.append(cause)

        if not causes:
            return "No root cause information found in the defect comments."

        lines = ["🔍 **Common Root Causes**\n"]
        cause_counter = Counter(causes)
        for cause, count in cause_counter.most_common(15):
            lines.append(f"  • {cause} ({count} occurrence(s))")

        lines.append(f"\n**Total defects with identified root cause:** {len(causes)} out of {len(self.df)}")
        return "\n".join(lines)

    def _severity_breakdown(self) -> str:
        """Breakdown of defects by severity."""
        col = 'Custom field (Severity)'
        if col not in self.df.columns:
            return "Severity data not available."

        counts = self.df[self.df[col].str.strip() != ''][col].value_counts()
        if counts.empty:
            return "No severity data found."

        lines = ["🔴 **Defect Severity Breakdown**\n"]
        total = counts.sum()
        for sev, count in counts.items():
            pct = count / total * 100
            bar = "█" * min(int(pct / 2), 30)
            lines.append(f"  **{sev}**: {count} ({pct:.0f}%) {bar}")
        return "\n".join(lines)

    def _assignee_analysis(self) -> str:
        """Defect distribution by assignee."""
        col = 'Assignee'
        if col not in self.df.columns:
            return "Assignee data not available."

        counts = self.df[self.df[col].str.strip() != ''][col].value_counts()
        if counts.empty:
            return "No assignee data found."

        lines = ["👤 **Defect Distribution by Assignee**\n"]
        for assignee, count in counts.items():
            lines.append(f"  • **{assignee}**: {count} defect(s)")
        return "\n".join(lines)

    def _release_analysis(self) -> str:
        """Defect distribution by release / MTP Gate."""
        col = 'Custom field (MTP Gate)'
        if col not in self.df.columns:
            return "Release data not available."

        counts = self.df[self.df[col].str.strip() != ''][col].value_counts()
        if counts.empty:
            return "No release data found."

        lines = ["🚀 **Defects by Release**\n"]
        for release, count in counts.items():
            lines.append(f"  • **{release}**: {count} defect(s)")
        return "\n".join(lines)

    def _status_breakdown(self) -> str:
        """Defect distribution by status."""
        col = 'Status'
        counts = self.df[self.df[col].str.strip() != ''][col].value_counts()
        if counts.empty:
            return "No status data found."

        lines = ["📋 **Defect Status Breakdown**\n"]
        total = counts.sum()
        for status, count in counts.items():
            pct = count / total * 100
            lines.append(f"  • **{status}**: {count} ({pct:.0f}%)")
        return "\n".join(lines)

    def _tcode_analysis(self) -> str:
        """Identify which transaction codes have the most defects."""
        tcode_counter: Counter = Counter()
        for _, row in self.df.iterrows():
            text = f"{row.get('Summary', '')} {row.get('Description', '')}"
            tcodes = re.findall(r'\b([A-Z]{2,4}\d{1,3}[A-Z]?)\b', text)
            for tc in tcodes:
                tcode_counter[tc] += 1

        if not tcode_counter:
            return "No TCode information found in defects."

        lines = ["💻 **Defects by Transaction Code**\n"]
        for tc, count in tcode_counter.most_common(15):
            lines.append(f"  • `{tc}`: {count} defect(s)")
        return "\n".join(lines)

    def _full_overview(self) -> str:
        """Comprehensive defect statistics overview."""
        df = self.df
        total = len(df)
        if total == 0:
            return "No defect data available."

        lines = [f"📊 **Defect Pattern Overview** ({total} total defects)\n"]

        # Status
        status_counts = df[df['Status'].str.strip() != '']['Status'].value_counts()
        if not status_counts.empty:
            lines.append("**By Status:**")
            for s, c in status_counts.items():
                lines.append(f"  • {s}: {c}")
            lines.append("")

        # Severity
        sev_col = 'Custom field (Severity)'
        if sev_col in df.columns:
            sev_counts = df[df[sev_col].str.strip() != ''][sev_col].value_counts()
            if not sev_counts.empty:
                lines.append("**By Severity:**")
                for s, c in sev_counts.items():
                    lines.append(f"  • {s}: {c}")
                lines.append("")

        # Top assignees
        if 'Assignee' in df.columns:
            asgn = df[df['Assignee'].str.strip() != '']['Assignee'].value_counts().head(5)
            if not asgn.empty:
                lines.append("**Top Assignees:**")
                for a, c in asgn.items():
                    lines.append(f"  • {a}: {c}")
                lines.append("")

        # Release distribution
        gate_col = 'Custom field (MTP Gate)'
        if gate_col in df.columns:
            rel = df[df[gate_col].str.strip() != ''][gate_col].value_counts().head(5)
            if not rel.empty:
                lines.append("**Top Releases:**")
                for r, c in rel.items():
                    lines.append(f"  • {r}: {c}")
                lines.append("")

        # Date range
        valid_dates = df.dropna(subset=['_created_dt'])
        if not valid_dates.empty:
            earliest = valid_dates['_created_dt'].min().strftime('%d/%m/%Y')
            latest = valid_dates['_created_dt'].max().strftime('%d/%m/%Y')
            lines.append(f"**Date Range:** {earliest} – {latest}")

        return "\n".join(lines)
