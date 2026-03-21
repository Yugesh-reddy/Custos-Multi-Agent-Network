"""Structured logging for co-evolution experiments."""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional


class EvolutionLogger:
    """JSONL logger for co-evolution generation data."""

    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = output_dir
        self._log_path = None
        if output_dir:
            path = Path(output_dir)
            path.mkdir(parents=True, exist_ok=True)
            self._log_path = path / "evolution_log.jsonl"

    def log_generation(self, gen_data: Dict):
        """Append a generation's results to the log."""
        if self._log_path:
            with open(self._log_path, "a") as f:
                f.write(json.dumps(gen_data) + "\n")

    def load_log(self) -> list:
        """Load all generation data from the log file."""
        if not self._log_path or not self._log_path.exists():
            return []
        entries = []
        with open(self._log_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
        return entries
