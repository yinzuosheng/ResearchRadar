import os
import smtplib
from email.mime.text import MIMEText

from utils.config import load_agent_config, load_tools_config
from utils.http import post_json
from utils.logger import logger


def push_message(message: str, channel: str = "") -> None:
    agent_cfg = load_agent_config().get("push", {})
    channel = channel or agent_cfg.get("channel", "feishu")

    if channel == "feishu":
        _push_feishu(message)
    elif channel == "dingtalk":
        _push_dingtalk(message)
    elif channel == "email":
        _push_email(message)
    else:
        raise ValueError(f"unsupported channel: {channel}")


def _push_feishu(message: str) -> None:
    cfg = load_tools_config().get("feishu", {})
    webhook = os.getenv(cfg.get("webhook_env", "FEISHU_WEBHOOK"), "")
    if not webhook:
        raise RuntimeError("Feishu webhook is missing")
    payload = {"msg_type": "text", "content": {"text": message}}
    post_json(webhook, payload)
    logger.info("feishu push done")


def _push_dingtalk(message: str) -> None:
    cfg = load_tools_config().get("dingtalk", {})
    webhook = os.getenv(cfg.get("webhook_env", "DINGTALK_WEBHOOK"), "")
    if not webhook:
        raise RuntimeError("DingTalk webhook is missing")
    payload = {"msgtype": "text", "text": {"content": message}}
    post_json(webhook, payload)
    logger.info("dingtalk push done")


def _push_email(message: str) -> None:
    cfg = load_tools_config().get("smtp", {})
    host = os.getenv(cfg.get("host_env", "SMTP_HOST"), "")
    port = int(os.getenv(cfg.get("port_env", "SMTP_PORT"), "465"))
    user = os.getenv(cfg.get("user_env", "SMTP_USER"), "")
    password = os.getenv(cfg.get("pass_env", "SMTP_PASS"), "")
    to_addr = os.getenv(cfg.get("to_env", "SMTP_TO"), "")

    msg = MIMEText(message, "plain")
    msg["Subject"] = "Daily Intelligence Brief"
    msg["From"] = user
    msg["To"] = to_addr

    with smtplib.SMTP_SSL(host, port) as server:
        server.login(user, password)
        server.sendmail(user, [to_addr], msg.as_string())
    logger.info("email push done")
