import os
import sys
import re
import json
import logging
from datetime import datetime, timedelta
from typing import List, Tuple, Optional

import requests
from bs4 import BeautifulSoup
from dateutil import parser as dtparse
import pytz
import yaml
import random
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

# --------- Required environment variables (set as GitHub Secrets) ----------
CONFLUENCE_BASE_URL = os.environ.get("CONFLUENCE_BASE_URL")   # e.g. https://your-domain.atlassian.net/wiki
CONFLUENCE_PAGE_ID  = os.environ.get("CONFLUENCE_PAGE_ID")    # numeric ID
CONFLUENCE_USER     = os.environ.get("CONFLUENCE_USER")       # Atlassian email
CONFLUENCE_API_TOKEN= os.environ.get("CONFLUENCE_API_TOKEN")  # Atlassian API token
SLACK_BOT_TOKEN     = os.environ.get("SLACK_BOT_TOKEN")       # xoxb-...
SLACK_CHANNEL_ID    = os.environ.get("SLACK_CHANNEL_ID")      # e.g. C0123456789 (or D…/G…)
SLACK_POST_TO_USER_ID = os.environ.get("SLACK_POST_TO_USER_ID")  # Optional: U… (opens DM via API; needs im:write)
PARSE_DUMP_ONLY = os.environ.get("PARSE_DUMP_ONLY", "false").lower() in {"1","true","yes"}

# Optional: change config path if you want
CONFIG_PATH         = os.environ.get("CONFIG_PATH", "config.yaml")

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("presenter-bot")


# ----------------- Config -----------------
class BotConfig:
    def __init__(self, data: dict):
        self.timezone = data.get("timezone", "Europe/Paris")

        cols = data.get("columns", {}) or {}
        self.col_date = cols.get("date", "date").strip().lower()
        self.col_presenter = cols.get("presenter", "presenter").strip().lower()
        # topic is parsed (to keep the table shape) but not used in messages
        self.col_topic = cols.get("topic", "topic").strip().lower()

        self.require_at = bool(data.get("require_presenter_at_mention", True))
        self.accept_multiple = bool(data.get("accept_multiple_in_week", False))

        # Title + templates
        self.title = data.get("title", ":peepo-chat: SUPER IMPORTANT REMINDER :peepo-chat:")
        self.presenter_templates = data.get("presenter_templates", [
            "Hope you all had a fabulous weekend! {presenter} is presenting on {date}",
            "Well well well… if it isn't the best team PAX has ever seen. Buckle up: {presenter} is presenting on {date}",
            "Heads up! {presenter} takes the mic on {date}",
            "Showtime: {presenter} on {date}",
        ])
        self.no_presenter_templates = data.get("no_presenter_templates", [
            "No one is presenting this week! Enjoy your freedom while it lasts 🎉",
            "Empty stage this week—use the time to recharge 🔋",
            "No talk this week. Coffee + deep work, anyone? ☕",
        ])
        # "rotate" | "random" (stable per week) | "random_per_run" (different every run)
        self.template_mode = data.get("template_mode", "rotate")

    @classmethod
    def load(cls, path: str) -> "BotConfig":
        with open(path, "r", encoding="utf-8") as f:
            return cls(yaml.safe_load(f) or {})


# ----------------- Guard -----------------
def require_env():
    missing = [k for k in [
        "CONFLUENCE_BASE_URL", "CONFLUENCE_PAGE_ID",
        "CONFLUENCE_USER", "CONFLUENCE_API_TOKEN",
        "SLACK_BOT_TOKEN"
    ] if not os.environ.get(k)]
    # SLACK_CHANNEL_ID or SLACK_POST_TO_USER_ID must exist
    if not os.environ.get("SLACK_CHANNEL_ID") and not os.environ.get("SLACK_POST_TO_USER_ID"):
        missing.append("SLACK_CHANNEL_ID or SLACK_POST_TO_USER_ID")
    if missing:
        raise RuntimeError(f"Missing env vars: {', '.join(missing)}")


# ----------------- Time helpers -----------------
def week_bounds(now: datetime, tzname: str) -> Tuple[datetime, datetime]:
    tz = pytz.timezone(tzname)
    local_now = now.astimezone(tz)
    monday = local_now - timedelta(days=local_now.weekday())
    start = monday.replace(hour=0, minute=0, second=0, microsecond=0)
    end = start + timedelta(days=6, hours=23, minutes=59, seconds=59, microseconds=999999)
    return start, end

def day_bounds(now: datetime, tzname: str) -> Tuple[datetime, datetime]:
    tz = pytz.timezone(tzname)
    local_now = now.astimezone(tz)
    start = local_now.replace(hour=0, minute=0, second=0, microsecond=0)
    end = start + timedelta(days=1, microseconds=-1)
    return start, end


# ----------------- Confluence fetch & parse -----------------
def fetch_confluence_storage_html() -> str:
    url = f"{CONFLUENCE_BASE_URL}/rest/api/content/{CONFLUENCE_PAGE_ID}?expand=body.storage,version"
    resp = requests.get(url, auth=(CONFLUENCE_USER, CONFLUENCE_API_TOKEN))
    if resp.status_code != 200:
        raise RuntimeError(f"Confluence API error {resp.status_code}: {resp.text[:400]}")
    body = resp.json().get("body", {}).get("storage", {}).get("value", "")
    if not body:
        raise RuntimeError("Confluence page has no storage body content.")
    return body

def parse_tables(html: str):
    """
    Returns: List[tables], each table is List[rows], each row is List[cells],
    each cell is a tuple (text, raw_html).
    """
    soup = BeautifulSoup(html, "html.parser")
    tables = []
    for tbl in soup.find_all("table"):
        rows = []
        for tr in tbl.find_all("tr"):
            cells = tr.find_all(["th", "td"])
            row = []
            for c in cells:
                txt = (c.get_text(" ", strip=True) or "").strip()
                row.append((txt, str(c)))
            if row:
                rows.append(row)
        if rows:
            tables.append(rows)
    return tables

def find_header_indexes(header_row: List[Tuple[str, str]], cfg: BotConfig) -> Optional[Tuple[int, int, Optional[int]]]:
    lower = [t.strip().lower() for (t, _h) in header_row]
    idx_date = idx_presenter = idx_topic = None
    for i, h in enumerate(lower):
        if h == cfg.col_date:
            idx_date = i
        elif h == cfg.col_presenter:
            idx_presenter = i
        elif h == cfg.col_topic:
            idx_topic = i
    if idx_date is None or idx_presenter is None:
        return None
    return idx_date, idx_presenter, idx_topic

def parse_date(s: str, tzname: str) -> Optional[datetime]:
    if not s:
        return None
    try:
        dt = dtparse.parse(s, dayfirst=False)
    except Exception:
        try:
            dt = dtparse.parse(s, dayfirst=True)
        except Exception:
            return None
    tz = pytz.timezone(tzname)
    if dt.tzinfo is None:
        dt = tz.localize(dt.replace(hour=0, minute=0, second=0, microsecond=0))
    else:
        dt = dt.astimezone(tz)
    return dt

def parse_confluence_date_from_html(cell_html: str, tzname: str) -> Optional[datetime]:
    """Use data-timestamp from <span data-node-type="date" data-timestamp="..."> if present."""
    soup = BeautifulSoup(cell_html or "", "html.parser")
    span = soup.find(attrs={"data-node-type": "date"})
    ts = span.get("data-timestamp") if span else None
    if not ts:
        return None
    try:
        ms = int(ts)
    except Exception:
        return None
    tz = pytz.timezone(tzname)
    return datetime.fromtimestamp(ms / 1000, tz=pytz.UTC).astimezone(tz).replace(
        hour=0, minute=0, second=0, microsecond=0
    )

def is_confluence_user_mention(cell_html: str) -> bool:
    h = (cell_html or "").lower()
    return ("ri:user" in h) or ("ri:account-id" in h) or ('data-linked-resource-type="user"' in h)


# ----------------- Schedule extraction -----------------
def extract_schedule(tables, cfg: BotConfig) -> List[Tuple[datetime, str]]:
    items: List[Tuple[datetime, str]] = []

    DEBUG_LOG = os.environ.get("DEBUG_LOG", "false").lower() in {"1","true","yes"}

    for t in tables:
        if not t:
            continue

        header = t[0]  # [(text, html), ...]
        idxs = find_header_indexes(header, cfg)
        if not idxs:
            continue
        idx_date, idx_presenter, _idx_topic = idxs

        for row in t[1:]:
            if max(idx_date, idx_presenter) >= len(row):
                continue

            (date_text, date_html) = row[idx_date]
            (presenter_text, presenter_html) = row[idx_presenter]

            date_text = (date_text or "").strip()
            presenter_text = (presenter_text or "").strip()

            # Accept either literal "@..." OR a real Confluence mention element
            literal_at = presenter_text.startswith("@")
            mention_elem = is_confluence_user_mention(presenter_html)

            if cfg.require_at and not (literal_at or mention_elem):
                if DEBUG_LOG:
                    log.info(f"Skip: presenter not a mention -> '{presenter_text}'")
                continue

            # Prefer Confluence timestamp; fallback to text parse
            dt = parse_confluence_date_from_html(date_html, cfg.timezone) or parse_date(date_text, cfg.timezone)
            if not dt:
                if DEBUG_LOG:
                    has_date_node = 'data-node-type="date"' in (date_html or "")
                    log.info(f"Skip: unparseable date -> '{date_text}' / html contains date-node={has_date_node}")
                continue

            # If it was a real mention but no visible "@", prefix for display
            if not presenter_text.startswith("@") and mention_elem:
                presenter_text = f"@{presenter_text}"

            items.append((dt, presenter_text))

    uniq = {(d.isoformat(), p): (d, p) for d, p in items}
    out = list(uniq.values())
    out.sort(key=lambda x: x[0])
    return out

def pick_for_week(schedule: List[Tuple[datetime, str]], start: datetime, end: datetime) -> List[Tuple[datetime, str]]:
    chosen = [(d, p) for (d, p) in schedule if start <= d <= end]
    chosen.sort(key=lambda x: x[0])
    return chosen


# ----------------- Templates -----------------
def pick_template(cfg: BotConfig, has_presenter: bool, now_local: datetime) -> str:
    pool = cfg.presenter_templates if has_presenter else cfg.no_presenter_templates
    if not pool:
        return ""

    mode = (cfg.template_mode or "rotate").lower()

    if mode == "random_per_run":
        return random.choice(pool)

    if mode == "random":
        # Stable across the same ISO week number
        random.seed(now_local.isocalendar().week)
        return random.choice(pool)

    # default: rotate by week number
    idx = (now_local.isocalendar().week - 1) % len(pool)
    return pool[idx]

def format_message(entries: List[Tuple[datetime, str]], cfg: BotConfig) -> str:
    tz = pytz.timezone(cfg.timezone)
    now_local = datetime.now(tz)

    if not entries:
        body = pick_template(cfg, has_presenter=False, now_local=now_local)
        return f"{cfg.title}\n\n{body}"

    # If multiple in one week, optionally list all or pick the earliest
    if not cfg.accept_multiple:
        entries = entries[:1]

    lines = []
    if len(entries) == 1:
        d, p = entries[0]
        date_str = d.strftime('%d %B %Y')  # e.g., 25 September 2025
        tmpl = pick_template(cfg, has_presenter=True, now_local=d)
        body = tmpl.format(presenter=p, date=date_str)
        lines.append(body)
    else:
        for d, p in entries:
            date_str = d.strftime('%d %B %Y')
            tmpl = pick_template(cfg, has_presenter=True, now_local=d)
            lines.append(tmpl.format(presenter=p, date=date_str))

    return f"{cfg.title}\n\n" + "\n".join(lines)


# ----------------- Slack -----------------
def resolve_channel_id(client: WebClient) -> str:
    """
    If SLACK_POST_TO_USER_ID is set (U…), open a DM and return its channel id (needs im:write).
    Otherwise use SLACK_CHANNEL_ID as-is (C…/G…/D…).
    """
    if SLACK_POST_TO_USER_ID:
        resp = client.conversations_open(users=SLACK_POST_TO_USER_ID)
        return resp["channel"]["id"]
    if not SLACK_CHANNEL_ID:
        raise RuntimeError("No SLACK_CHANNEL_ID or SLACK_POST_TO_USER_ID provided.")
    return SLACK_CHANNEL_ID

def post_to_slack(text: str):
    client = WebClient(token=SLACK_BOT_TOKEN)
    channel = resolve_channel_id(client)
    try:
        client.chat_postMessage(channel=channel, text=text)
    except SlackApiError as e:
        raise RuntimeError(f"Slack API error: {e.response.get('error')}")


# ----------------- Diagnostics (optional JSON artifact) -----------------
def dump_parse_diagnostics(tables, cfg, limit=200, out_path=None):
    rows = []
    tz = pytz.timezone(cfg.timezone)

    for t in tables:
        if not t:
            continue

        header_texts = [txt for (txt, _h) in t[0]]
        idxs = find_header_indexes(t[0], cfg)
        idx_date = idx_presenter = None
        if idxs:
            idx_date, idx_presenter, _ = idxs
        log.info(f"[diag] header={header_texts} idx_date={idx_date} idx_presenter={idx_presenter}")

        for r in t[1:]:
            if idx_date is None or idx_presenter is None or max(idx_date, idx_presenter) >= len(r):
                continue

            (date_text, date_html) = r[idx_date]
            (presenter_text, presenter_html) = r[idx_presenter]

            date_text = (date_text or "").strip()
            presenter_text = (presenter_text or "").strip()

            m = re.search(r'data-timestamp="(\d+)"', date_html or "", flags=re.I)
            ts_ms = int(m.group(1)) if m else None
            parsed_dt = parse_confluence_date_from_html(date_html, cfg.timezone) or parse_date(date_text, cfg.timezone)

            row_info = {
                "date_text": date_text,
                "has_date_node": bool(m),
                "date_ts_ms": ts_ms,
                "parsed_date_iso": parsed_dt.isoformat() if parsed_dt else None,
                "presenter_text": presenter_text,
                "literal_at": presenter_text.startswith("@"),
                "mention_elem": is_confluence_user_mention(presenter_html),
            }
            rows.append(row_info)
            log.info(
                f"[diag] row -> date_text='{row_info['date_text']}', ts_ms={row_info['date_ts_ms']}, "
                f"parsed={row_info['parsed_date_iso']}, presenter='{row_info['presenter_text']}', "
                f"literal@={row_info['literal_at']}, mention={row_info['mention_elem']}"
            )

            if len(rows) >= limit:
                break
        if len(rows) >= limit:
            break

    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)

# If missing in your file, include this:
def extract_account_ids(cell_html: str):
    import re
    return re.findall(r'ri:user[^>]+ri:account-id="([^"]+)"', cell_html or "", flags=re.I)

def dump_full_tables(tables, cfg, out_path="/tmp/confluence_tables_dump.json"):
    """Dump every cell's text/html and parsing signals to a JSON file."""
    payload = []
    for ti, t in enumerate(tables):
        if not t:
            continue
        header_texts = [txt for (txt, _h) in t[0]]
        table_entry = {"table_index": ti, "header": header_texts, "rows": []}

        for ri, row in enumerate(t[1:], start=1):
            row_entry = []
            for ci, (text, html) in enumerate(row):
                # Signals for debugging
                date_from_html = parse_confluence_date_from_html(html, cfg.timezone)
                date_from_text = parse_date(text, cfg.timezone)
                row_entry.append({
                    "col_index": ci,
                    "text": text,
                    "html_snippet": (html or "")[:600],  # trim to keep artifact manageable
                    "has_date_node": ('data-node-type="date"' in (html or "")),
                    "date_ts_ms": (
                        int(__import__("re").search(r'data-timestamp="(\d+)"', html or "", flags=__import__("re").I).group(1))
                        if __import__("re").search(r'data-timestamp="(\d+)"', html or "", flags=__import__("re").I) else None
                    ),
                    "parsed_date_iso_html": date_from_html.isoformat() if date_from_html else None,
                    "parsed_date_iso_text": date_from_text.isoformat() if date_from_text else None,
                    "mention_elem": is_confluence_user_mention(html),
                    "account_ids": extract_account_ids(html),
                    "literal_at": (text or "").strip().startswith("@"),
                })
            table_entry["rows"].append(row_entry)
        payload.append(table_entry)

    import json
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    log.info(f"[dump] wrote {out_path} (tables={len(payload)})")



# ----------------- Main -----------------
def main():
    require_env()
    cfg = BotConfig.load(CONFIG_PATH)

    # Env toggles for testing / debugging
    DEBUG_LOG = os.environ.get("DEBUG_LOG", "false").lower() in {"1", "true", "yes"}
    TEST_TODAY_ONLY = os.environ.get("TEST_TODAY_ONLY", "false").lower() in {"1", "true", "yes"}
    FORCE_DATE_STR = (os.environ.get("FORCE_DATE") or "").strip()  # e.g., "2025-09-25"

    # Establish 'now' in the configured timezone
    tz = pytz.timezone(cfg.timezone)
    now = datetime.now(tz)

    # If forcing a specific date (useful for reproducing a known row)
    if FORCE_DATE_STR:
        forced = dtparse.parse(FORCE_DATE_STR)
        if forced.tzinfo is None:
            forced = tz.localize(forced)
        else:
            forced = forced.astimezone(tz)
        now = forced

    # Select the time window
    if TEST_TODAY_ONLY or FORCE_DATE_STR:
        start, end = day_bounds(now, cfg.timezone)
        log.info(f"Today-only window: {start.date()} [{cfg.timezone}]")
    else:
        start, end = week_bounds(now, cfg.timezone)
        log.info(f"Week window: {start.date()} → {end.date()} [{cfg.timezone}]")

    # Fetch + parse Confluence page
    html = fetch_confluence_storage_html()
    tables = parse_tables(html)  # cells as (text, html)

    if PARSE_DUMP_ONLY:
        dump_full_tables(tables, cfg, out_path="/tmp/confluence_tables_dump.json")
        log.info("Dump-only mode: wrote /tmp/confluence_tables_dump.json and exited.")
        return

    # Optional: diagnostics to logs + JSON artifact
    if DEBUG_LOG:
        dump_parse_diagnostics(tables, cfg, limit=300, out_path="/tmp/presenter_parse_debug.json")

    # Build schedule from tables (handles date-node + @mentions)
    schedule = extract_schedule(tables, cfg)

    if DEBUG_LOG:
        log.info(f"Total parsed rows kept: {len(schedule)}")
        for d, p in schedule[:50]:
            log.info(f"[diag] kept -> {p} @ {d.strftime('%Y-%m-%d')}")

    # Filter to window (today or week)
    entries = pick_for_week(schedule, start, end)

    if DEBUG_LOG:
        log.info(f"Entries in window: {len(entries)}")
        for d, p in entries:
            log.info(f"[diag] window -> {p} @ {d.strftime('%Y-%m-%d')}")

    # Format + send to Slack
    message = format_message(entries, cfg)
    post_to_slack(message)
    log.info("Posted to Slack.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log.error(str(e))
        sys.exit(1)
