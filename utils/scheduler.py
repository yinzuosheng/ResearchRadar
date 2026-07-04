from apscheduler.schedulers.blocking import BlockingScheduler

from utils.config import load_agent_config
from utils.logger import logger
from utils.pipeline import run_daily_brief


def start_scheduler() -> None:
    cfg = load_agent_config().get("schedule", {})
    if not cfg.get("enabled", True):
        logger.info("schedule is disabled")
        return

    time_str = cfg.get("time", "08:00")
    timezone = cfg.get("timezone", "Asia/Shanghai")
    hour, minute = [int(part) for part in time_str.split(":", maxsplit=1)]

    scheduler = BlockingScheduler(timezone=timezone)
    scheduler.add_job(run_daily_brief, "cron", hour=hour, minute=minute)
    logger.info("scheduler started %s %s", hour, minute)
    scheduler.start()
from apscheduler.schedulers.blocking import BlockingScheduler

from utils.config import load_agent_config
from utils.logger import logger
from utils.pipeline import run_daily_brief


def start_scheduler() -> None:
    cfg = load_agent_config().get("schedule", {})
    if not cfg.get("enabled", True):
        logger.info("schedule is disabled")
        return

    time_str = cfg.get("time", "08:00")
    timezone = cfg.get("timezone", "Asia/Shanghai")
    hour, minute = [int(part) for part in time_str.split(":", maxsplit=1)]

    scheduler = BlockingScheduler(timezone=timezone)
    scheduler.add_job(run_daily_brief, "cron", hour=hour, minute=minute)
    logger.info("scheduler started %s %s", hour, minute)
    scheduler.start()
