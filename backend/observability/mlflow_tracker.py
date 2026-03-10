from __future__ import annotations

import hashlib
import json
import os
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

from backend.settings import settings

try:
    import mlflow
    from mlflow.tracking import MlflowClient
except Exception:  # pragma: no cover - optional dependency
    mlflow = None
    MlflowClient = None


def _sha1_text(text: str) -> str:
    return hashlib.sha1((text or "").encode("utf-8", errors="ignore")).hexdigest()[:12]


@dataclass
class PromptSpec:
    name: str
    text: str
    tool_name: str
    description: str = ""


class MLflowTracker:
    def __init__(self) -> None:
        self.enabled = bool(settings.MLFLOW_ENABLED and mlflow and settings.MLFLOW_TRACKING_URI)
        self.client = None
        if not self.enabled:
            return

        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
        self.client = MlflowClient()

    def ensure_experiment(self, name: str) -> Optional[str]:
        if not self.enabled or not self.client:
            return None
        exp = self.client.get_experiment_by_name(name)
        if exp:
            return exp.experiment_id
        return self.client.create_experiment(
            name=name,
            artifact_location=settings.MLFLOW_ARTIFACT_LOCATION or None,
        )

    @contextmanager
    def start_run(
        self,
        run_name: str,
        experiment_name: Optional[str] = None,
        tags: Optional[Dict[str, Any]] = None,
        nested: bool = False,
    ) -> Iterator[None]:
        if not self.enabled:
            yield
            return

        experiment_id = self.ensure_experiment(experiment_name or settings.MLFLOW_EXPERIMENT)
        with mlflow.start_run(run_name=run_name, experiment_id=experiment_id, nested=nested):
            safe_tags = {k: str(v) for k, v in (tags or {}).items()}
            safe_tags.setdefault("environment", settings.MLFLOW_TAG_ENV)
            mlflow.set_tags(safe_tags)
            yield

    def log_params(self, params: Dict[str, Any]) -> None:
        if not self.enabled or not params:
            return
        for key, value in params.items():
            mlflow.log_param(key, value if isinstance(value, (str, int, float, bool)) else json.dumps(value, ensure_ascii=False))

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        if not self.enabled or not metrics:
            return
        for key, value in metrics.items():
            if isinstance(value, bool):
                value = 1.0 if value else 0.0
            if isinstance(value, (int, float)):
                mlflow.log_metric(key, float(value), step=step)

    def log_dict(self, payload: Dict[str, Any], artifact_file: str) -> None:
        if not self.enabled:
            return
        mlflow.log_dict(payload, artifact_file)

    def log_text(self, text: str, artifact_file: str) -> None:
        if not self.enabled:
            return
        mlflow.log_text(text, artifact_file)

    def log_artifacts(self, paths: Iterable[str], artifact_path: Optional[str] = None) -> None:
        if not self.enabled:
            return
        for path in paths:
            if path and os.path.exists(path):
                mlflow.log_artifact(path, artifact_path=artifact_path)

    def log_directory(self, path: str, artifact_path: Optional[str] = None) -> None:
        if not self.enabled or not path or not os.path.exists(path):
            return
        mlflow.log_artifacts(path, artifact_path=artifact_path)

    def log_prompt_registry(self, prompts: List[PromptSpec]) -> Dict[str, str]:
        versions: Dict[str, str] = {}
        if not prompts:
            return versions

        with self.start_run(
            run_name="prompt-registry-sync",
            experiment_name=settings.MLFLOW_PROMPT_EXPERIMENT,
            tags={"kind": "prompt_registry"},
        ):
            for prompt in prompts:
                version = _sha1_text(prompt.text)
                versions[prompt.name] = version
                self.log_params({
                    f"prompt.{prompt.name}.tool": prompt.tool_name,
                    f"prompt.{prompt.name}.version": version,
                })
                payload = {
                    "name": prompt.name,
                    "tool_name": prompt.tool_name,
                    "version": version,
                    "description": prompt.description,
                    "text": prompt.text,
                }
                self.log_dict(payload, f"prompts/{prompt.name}/{version}.json")
        return versions

    def log_pipeline_snapshot(self, manifest: Dict[str, Any], files: Optional[List[str]] = None) -> None:
        with self.start_run(
            run_name="pipeline-snapshot",
            experiment_name=settings.MLFLOW_EXPERIMENT,
            tags={"kind": "pipeline_registry", "pipeline_name": settings.MLFLOW_PIPELINE_NAME},
        ):
            self.log_dict(manifest, "pipeline/manifest.json")
            self.log_artifacts(files or [], artifact_path="pipeline/files")

    def write_text_artifact(self, text: str, suffix: str = ".txt") -> str:
        fd, path = tempfile.mkstemp(suffix=suffix)
        os.close(fd)
        Path(path).write_text(text, encoding="utf-8")
        return path


mlflow_tracker = MLflowTracker()
