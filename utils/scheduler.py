"""Failure-safe weekly research workflow and scheduler wiring."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from uuid import uuid4

from apscheduler.schedulers.blocking import BlockingScheduler

from domain.models import TrendReport, WeeklyWorkflowResult
from utils.config import load_agent_config
from utils.logger import logger


def configure_weekly_job(scheduler, job, schedule: dict) -> None:
    hour, minute = (int(part) for part in schedule.get("time", "08:00").split(":", 1))
    timezone = schedule.get("timezone", "Asia/Shanghai")
    scheduler.add_job(job, "cron", day_of_week="mon", hour=hour, minute=minute, timezone=timezone)


class WeeklyWorkflow:
    def __init__(self, *, sync, generate_trend, reports_dir: Path | str, push=None) -> None:
        self.sync = sync
        self.generate_trend = generate_trend
        self.reports_dir = Path(reports_dir).resolve()
        self.push = push

    def run(self, since: datetime, *, push_enabled: bool = False) -> WeeklyWorkflowResult:
        try:
            self.sync()
        except Exception:
            return WeeklyWorkflowResult(status="sync_failed", error="weekly_sync_failed")
        try:
            report = self.generate_trend(since)
        except Exception:
            return WeeklyWorkflowResult(status="trend_failed", error="weekly_trend_failed")
        try:
            saved = self._save(report)
        except Exception:
            return WeeklyWorkflowResult(status="save_failed", error="weekly_save_failed")
        if push_enabled and self.push is not None:
            try:
                self.push(report.model_dump_json(indent=2))
            except Exception:
                return WeeklyWorkflowResult(status="delivery_failed", saved_path=str(saved), retryable=True, error="weekly_delivery_failed")
            return WeeklyWorkflowResult(status="delivered", saved_path=str(saved))
        return WeeklyWorkflowResult(status="saved", saved_path=str(saved))

    def _save(self, report: TrendReport) -> Path:
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        filename = f"trend-{report.generated_at:%Y%m%d-%H%M%S}-{uuid4().hex[:8]}.json"
        target = (self.reports_dir / filename).resolve()
        if self.reports_dir not in target.parents:
            raise ValueError("weekly_report_path_invalid")
        temporary = target.with_suffix(target.suffix + ".part")
        try:
            temporary.write_text(report.model_dump_json(indent=2), encoding="utf-8")
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
        return target


def start_scheduler(job=None) -> None:
    cfg = load_agent_config().get("schedule", {})
    if not cfg.get("enabled", True):
        logger.info("schedule is disabled")
        return
    if job is None:
        raise ValueError("weekly_job_required")
    scheduler = BlockingScheduler(timezone=cfg.get("timezone", "Asia/Shanghai"))
    configure_weekly_job(scheduler, job, cfg)
    scheduler.start()
