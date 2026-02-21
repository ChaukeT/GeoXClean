"""
InSAR Controller (ISCE-2 External Orchestration).

Responsible for generating ISCE configs, launching jobs via WSL2, and
registering output artefacts in the registry for audit-safe ingestion.
"""

from __future__ import annotations

import json
import logging
import shlex
import subprocess
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..core.data_registry_simple import DataRegistrySimple

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class InsarJobSpec:
    job_id: str
    workflow: str
    output_dir: str
    wsl_distro: Optional[str]
    command: str
    inputs: Dict[str, Any]
    parameters: Dict[str, Any]
    created_at: str


@dataclass(frozen=True)
class InsarOutput:
    key: str
    path: str
    artifact_type: str
    status: str = "registered"


class WslRunner:
    """Minimal WSL2 command runner for ISCE-2 jobs."""

    def run(self, command: str, distro: Optional[str] = None, cwd: Optional[Path] = None) -> Dict[str, Any]:
        if not command:
            raise ValueError("No command specified for WSL runner")

        wsl_cmd = ["wsl.exe"]
        if distro:
            wsl_cmd += ["-d", distro]

        # Sanitize command and path to prevent command injection
        bash_cmd = shlex.quote(command)
        if cwd:
            bash_cmd = f"cd {shlex.quote(cwd.as_posix())} && {shlex.quote(command)}"

        wsl_cmd += ["--", "bash", "-lc", bash_cmd]
        logger.info("WSL run (sanitized): %s", " ".join(wsl_cmd))

        result = subprocess.run(
            wsl_cmd,
            capture_output=True,
            text=True,
            check=False,
        )
        return {
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }


class InsarController:
    """Controller for external ISCE-2 InSAR workflows."""

    OUTPUT_KEYS = {
        "index": "insar_outputs_index",
        "los": "insar_los_raster",
        "velocity": "insar_velocity_map",
        "coherence": "insar_coherence_map",
        "timeseries": "insar_timeseries",
    }

    def __init__(self, app_controller: Any):
        self._app = app_controller
        self.registry: DataRegistrySimple = app_controller.registry
        self._runner = WslRunner()
        self._latest_outputs: List[InsarOutput] = []

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def run_job(self, params: Dict[str, Any]) -> Dict[str, Any]:
        job_spec, job_dir = self._build_job_spec(params)
        self._write_job_spec(job_spec, job_dir)
        self._write_isce_config(job_spec, job_dir)

        run_result = self._runner.run(job_spec.command, distro=job_spec.wsl_distro, cwd=job_dir)
        outputs = self._register_outputs(job_spec, params)

        summary = "ISCE-2 job completed"
        if run_result["returncode"] != 0:
            summary = "ISCE-2 job failed (see stderr)"

        return {
            "summary": summary,
            "job_spec": asdict(job_spec),
            "outputs": [asdict(o) for o in outputs],
            "stdout": run_result.get("stdout", ""),
            "stderr": run_result.get("stderr", ""),
            "returncode": run_result.get("returncode", -1),
        }

    def pause_job(self):
        logger.info("Pause requested (not implemented).")

    def resume_job(self):
        logger.info("Resume requested (not implemented).")

    def list_results(self) -> List[Dict[str, str]]:
        results = []
        for output in self._latest_outputs:
            results.append(
                {
                    "type": output.artifact_type,
                    "key": output.key,
                    "path": output.path,
                    "status": output.status,
                }
            )
        return results

    def ingest_results(self, output_dir: str, params: Optional[Dict[str, Any]] = None) -> List[InsarOutput]:
        if not output_dir:
            raise ValueError("Output directory is required to ingest results")
        job_spec = InsarJobSpec(
            job_id=datetime.utcnow().strftime("insar_ingest_%Y%m%d_%H%M%S"),
            workflow=params.get("workflow") if params else "unknown",
            output_dir=output_dir,
            wsl_distro=params.get("wsl_distro") if params else None,
            command="ingest_only",
            inputs={},
            parameters=params or {},
            created_at=datetime.utcnow().isoformat(),
        )
        outputs = self._register_outputs(job_spec, params or {})
        return outputs

    def export_result(self, source_path: str, dest_path: str):
        source = Path(source_path)
        dest = Path(dest_path)
        if not source.exists():
            raise FileNotFoundError(f"Source file not found: {source_path}")
        dest.write_bytes(source.read_bytes())

    # ------------------------------------------------------------------ #
    # Job registry payload wrappers
    # ------------------------------------------------------------------ #
    def _prepare_insar_run_payload(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return self.run_job(params)

    def _prepare_insar_ingest_payload(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        output_dir = params.get("output_dir")
        outputs = self.ingest_results(output_dir, params)
        return [asdict(o) for o in outputs]

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _build_job_spec(self, params: Dict[str, Any]) -> tuple[InsarJobSpec, Path]:
        project_root = self._resolve_project_root()
        job_id = datetime.utcnow().strftime("insar_%Y%m%d_%H%M%S")
        job_dir = project_root / "insar_runs" / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        workflow = params.get("workflow") or "stripmapApp"
        output_dir = params.get("output_dir") or str(job_dir / "outputs")
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        command = params.get("command") or f"isce2 {workflow}"
        created_at = datetime.utcnow().isoformat()

        job_spec = InsarJobSpec(
            job_id=job_id,
            workflow=workflow,
            output_dir=output_dir,
            wsl_distro=params.get("wsl_distro"),
            command=command,
            inputs=self._collect_inputs(params),
            parameters=self._collect_parameters(params),
            created_at=created_at,
        )
        return job_spec, job_dir

    def _write_job_spec(self, job_spec: InsarJobSpec, job_dir: Path) -> None:
        spec_path = job_dir / "geox_insar_job.json"
        spec_path.write_text(json.dumps(asdict(job_spec), indent=2, sort_keys=True), encoding="utf-8")

    def _write_isce_config(self, job_spec: InsarJobSpec, job_dir: Path) -> None:
        config_path = job_dir / "isce_params.json"
        payload = {
            "workflow": job_spec.workflow,
            "inputs": job_spec.inputs,
            "parameters": job_spec.parameters,
            "output_dir": job_spec.output_dir,
        }
        config_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def _register_outputs(self, job_spec: InsarJobSpec, params: Dict[str, Any]) -> List[InsarOutput]:
        outputs: List[InsarOutput] = []
        output_dir = Path(job_spec.output_dir)

        def register(key: str, path: Path, artifact_type: str):
            meta = {
                "labels": ["RESULT"],
                "source_file": str(path),
                "method": f"insar_{artifact_type}",
                "job_id": job_spec.job_id,
            }
            if self.registry.register_model(key, {"path": str(path), "type": artifact_type}, metadata=meta, source_panel="insar"):
                outputs.append(InsarOutput(key=key, path=str(path), artifact_type=artifact_type))

        candidates = {
            "los": params.get("los_path") or output_dir / "los_displacement.tif",
            "velocity": params.get("velocity_path") or output_dir / "velocity_map.tif",
            "coherence": params.get("coherence_path") or output_dir / "coherence_map.tif",
            "timeseries": params.get("timeseries_path") or output_dir / "timeseries.csv",
        }

        for artifact_type, path in candidates.items():
            path = Path(path)
            register(self.OUTPUT_KEYS[artifact_type], path, artifact_type)

        index_key = self.OUTPUT_KEYS["index"]
        meta = {
            "labels": ["RESULT_INDEX"],
            "method": "insar_outputs_index",
            "job_id": job_spec.job_id,
        }
        index_payload = [asdict(o) for o in outputs]
        self.registry.register_model(index_key, index_payload, metadata=meta, source_panel="insar")

        self._latest_outputs = outputs
        return outputs

    def _collect_inputs(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "aoi_mode": params.get("aoi_mode"),
            "aoi_polygon": params.get("aoi_polygon"),
            "aoi_bbox": params.get("aoi_bbox"),
            "dem_path": params.get("dem_path"),
            "orbit_source": params.get("orbit_source"),
            "processing_mode": params.get("processing_mode"),
            "sentinel_products": params.get("sentinel_products", []),
        }

    def _collect_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        reserved = {"aoi_mode", "aoi_polygon", "aoi_bbox", "dem_path", "orbit_source", "processing_mode",
                    "sentinel_products", "workflow", "output_dir", "wsl_distro", "command"}
        return {k: v for k, v in params.items() if k not in reserved}

    def _resolve_project_root(self) -> Path:
        project_path = getattr(self._app.s, "project_path", None)
        if project_path:
            return Path(project_path)
        return Path.cwd()

