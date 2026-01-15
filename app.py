import os
import sys
import re
import json
import logging
import unicodedata
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
CONFLUENCE_BASE_URL   = os.environ.get("CONFLUENCE_BASE_URL")     # e.g. https://your-domain.atlassian.net/wiki
CONFLUENCE_PAGE_ID    = os.environ.get("CONFLUENCE_PAGE_ID")      # numeric ID
CONFLUENCE_USER       = os.environ.get("CONFLUENCE_USER")         # Atlassian email
CONFLUENCE_API_TOKEN  = os.environ.get("CONFLUENCE_API_TOKEN")    # Atlassian API token
SLACK_BOT_TOKEN       = os.environ.get("SLACK_BOT_TOKEN")         # xoxb-...
SLACK_CHANNEL_ID      = os.environ.get("SLACK_CHANNEL_ID")        # C…/G…/D…
SLACK_POST_TO_USER_ID = os.environ.get("SLACK_POST_TO_USER_ID")   # Optional U… (requires im:write)
PARSE_DUMP_ONLY       = os.environ.get("PARSE_DUMP_ONLY", "false").lower() in {"1", "true", "yes"}

# Optional: change config path if you want
CONFIG_PATH = os.environ.get("CONFIG_PATH", "config.yaml")

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("presenter-bot")


# ----------------- Small utils -----------------
def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s or "") if not unicodedata.combining(c))

def _norm_key(s: str) -> str:
    s = _strip_accents(s).lower().strip()
    return re.sub(r"\s+", " ", s)


# ----------------- Config -----------------
class BotConfig:
    def __init__(self, data: dict):
        self.timezone = data.get("timezone", "Europe/Paris")

        cols = data.get("columns", {}) or {}
        self.col_date = cols.get("date", "date").strip().lower()
        self.col_presenter = cols.get("presenter", "presenter").strip().lower()
        self.col_topic = cols.get("topic", "topic").strip().lower()  # parsed but unused

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
        # "rotate" | "random" (stable per week) | "random_per_run"
        self.template_mode = data.get("template_mode", "rotate")

        # Slack user map (normalized-name -> Slack user ID)
        self.slack_user_map = {}
        for k, v in (data.get("slack_user_map", {}) or {}).items():
            if v:
                self.slack_user_map[_norm_key(k)] = str(v).strip()

        # ----------------- Pairing config -----------------
        pairing = data.get("pairing", {}) or {}
        self.pairing_enabled = bool(pairing.get("enabled", True))
        self.pairing_header = pairing.get("header", ":handshake: 1:1 buddy for the week")
        self.pairing_line_template = pairing.get("line_template", "{a} ↔ {b}")
        self.pairing_state_path = pairing.get(
            "state_path",
            os.environ.get("PAIRING_STATE_PATH", ".pairing_state.json"),
        )

        # Pairing users ALWAYS come from slack_user_map
        self.pairing_user_ids: List[str] = list(dict.fromkeys(self.slack_user_map.values()))

        if len(self.pairing_user_ids) < 2:
            log.warning("[pairing] Not enough users to generate 1:1 pairs")


    @classmethod
    def load(cls, path: str) -> "BotConfig":
        with open(path, "r", encoding="utf-8") as f:
            return cls(yaml.safe_load(f) or {})


def slackify_presenter(label: str, cfg: BotConfig) -> str:
    """
    If label looks like '@Name', try to map it to a Slack mention <@U...>.
    Falls back to the original label if not found.
    """
    if not label or not label.startswith("@"):
        return label
    # Already a real mention?
    if re.fullmatch(r"<@[^>]+>", label):
        return label
    key = _norm_key(label[1:].strip())
    user_id = cfg.slack_user_map.get(key)
    return f"<@{user_id}>" if user_id else label


# ----------------- Guards -----------------
def require_env():
    missing = [k for k in [
        "CONFLUENCE_BASE_URL", "CONFLUENCE_PAGE_ID",
        "CONFLUENCE_USER", "CONFLUENCE_API_TOKEN",
        "SLACK_BOT_TOKEN"
    ] if not os.environ.get(k)]
    if not (os.environ.get("SLACK_CHANNEL_ID") or os.environ.get("SLACK_POST_TO_USER_ID")):
        missing.append("SLACK_CHANNEL_ID or SLACK_POST_TO_USER_ID")
    if missing:
        raise RuntimeError(f"Missing env vars: {', '.join(missing)}")


# ----------------- Time helpers -----------------
def week_bounds(now: datetime, tzname: str) -> Tuple[datetime, datetime]:
    tz = pytz.timezone(tzname)
    local_now = now.astimezone(tz)
    monday = local_now - timedelta(days=local_now.weekday())
    start = monday.replace(hour=0, minute=0, second=0, microsecond=0)
    end = start + timedelta(days=6, hours=23, minutes=59, seconds=59, microseconds=999_999)
    return start, end

def day_bounds(now: datetime, tzname: str) -> Tuple[datetime, datetime]:
    tz = pytz.timezone(tzname)
    local_now = now.astimezone(tz)
    start = local_now.replace(hour=0, minute=0, second=0, microsecond=0)
    end = start + timedelta(days=1, microseconds=-1)
    return start, end


# ----------------- Confluence helpers -----------------
def fetch_confluence_storage_html() -> str:
    url = f"{CONFLUENCE_BASE_URL}/rest/api/content/{CONFLUENCE_PAGE_ID}?expand=body.storage,version"
    resp = requests.get(url, auth=(CONFLUENCE_USER, CONFLUENCE_API_TOKEN), timeout=30)
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
    """Return (date_idx, presenter_idx, topic_idx?) when headers match config (case-insensitive)."""
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
    dt = dt.astimezone(tz) if dt.tzinfo else tz.localize(dt.replace(hour=0, minute=0, second=0, microsecond=0))
    return dt

def parse_confluence_date_from_html(cell_html: str, tzname: str) -> Optional[datetime]:
    """
    Parse Confluence date either from <time datetime='YYYY-MM-DD'> or
    from <span data-node-type='date' data-timestamp='...'>.
    """
    soup = BeautifulSoup(cell_html or "", "html.parser")
    tz = pytz.timezone(tzname)

    # <time datetime="2025-09-25">
    t = soup.find("time")
    if t and t.get("datetime"):
        try:
            d = dtparse.parse(t["datetime"])
            d = d.astimezone(tz) if d.tzinfo else tz.localize(d.replace(hour=0, minute=0, second=0, microsecond=0))
            return d
        except Exception:
            pass

    # <span data-node-type="date" data-timestamp="...">
    span = soup.find(attrs={"data-node-type": "date"})
    ts = span.get("data-timestamp") if span else None
    if ts:
        try:
            ms = int(ts)
            return datetime.fromtimestamp(ms / 1000, tz=pytz.UTC).astimezone(tz).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
        except Exception:
            return None

    return None

def is_confluence_user_mention(cell_html: str) -> bool:
    h = (cell_html or "").lower()
    return ("ri:user" in h) or ("ri:account-id" in h) or ('data-linked-resource-type="user"' in h)

USER_CACHE: dict = {}

def extract_account_ids(cell_html: str) -> List[str]:
    return re.findall(r'ri:user[^>]+ri:account-id="([^"]+)"', cell_html or "", flags=re.I)

def lookup_user_name(account_id: str) -> Optional[str]:
    if account_id in USER_CACHE:
        return USER_CACHE[account_id]
    url = f"{CONFLUENCE_BASE_URL}/rest/api/user"
    try:
        r = requests.get(url, params={"accountId": account_id}, auth=(CONFLUENCE_USER, CONFLUENCE_API_TOKEN), timeout=20)
        if r.status_code == 200:
            data = r.json()
            name = data.get("displayName") or data.get("publicName")
            if name:
                USER_CACHE[account_id] = name
                return name
    except Exception as e:
        log.info(f"[diag] user lookup failed for {account_id}: {e}")
    return None

def presenter_label_from_cell(presenter_text: str, presenter_html: str, require_at: bool) -> Optional[str]:
    """Return '@Display Name' if possible (resolves mention macros)."""
    presenter_text = (presenter_text or "").strip()

    if presenter_text:
        return presenter_text if (not require_at or presenter_text.startswith("@")) else f"@{presenter_text}"

    ids = extract_account_ids(presenter_html)
    if ids:
        name = lookup_user_name(ids[0])
        if name:
            return f"@{name}"

    # Optional fallback: some instances include an alias attribute.
    soup = BeautifulSoup(presenter_html or "", "html.parser")
    alias = soup.find(attrs={"data-linked-resource-default-alias": True})
    if alias:
        val = alias.get("data-linked-resource-default-alias")
        if val:
            return f"@{val}"

    return None


# ----------------- Schedule extraction -----------------
def extract_schedule(tables, cfg: BotConfig) -> List[Tuple[datetime, str]]:
    """
    Build (date, presenter_label) from Confluence tables.
    - Dates: prefer HTML (<time datetime> or data-timestamp) then fallback to text.
    - Presenters: accept literal '@...' OR mention macro; resolve accountId -> display name.
    """
    DEBUG_LOG = os.environ.get("DEBUG_LOG", "false").lower() in {"1", "true", "yes"}

    def infer_indexes_from_table(body_rows: List[List[Tuple[str, str]]]) -> Optional[Tuple[int, int, Optional[int]]]:
        """Heuristics when headers don't match: score columns for date/presenter signals."""
        if not body_rows:
            return None
        max_cols = max((len(r) for r in body_rows if r), default=0)
        if max_cols == 0:
            return None

        date_scores = [0] * max_cols
        presenter_scores = [0] * max_cols
        date_like_re = re.compile(r"\b(\d{1,2}[-/]\d{1,2}([-/]\d{2,4})?|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b", re.I)

        for r in body_rows[:50]:
            for ci in range(min(len(r), max_cols)):
                text, html = r[ci]
                h = (html or "").lower()
                t = (text or "")

                # date signals
                if "<time" in h and 'datetime="' in h:
                    date_scores[ci] += 4
                if 'data-node-type="date"' in h or "date-node" in h:
                    date_scores[ci] += 3
                if re.search(r'data-timestamp="\d+"', h):
                    date_scores[ci] += 3
                if date_like_re.search(t):
                    date_scores[ci] += 1

                # presenter signals
                if ("ri:user" in h) or ("ri:account-id" in h) or ('data-linked-resource-type="user"' in h):
                    presenter_scores[ci] += 3
                if t.strip().startswith("@"):
                    presenter_scores[ci] += 1

        if max(date_scores) == 0 or max(presenter_scores) == 0:
            return None
        return date_scores.index(max(date_scores)), presenter_scores.index(max(presenter_scores)), None

    items: List[Tuple[datetime, str]] = []

    for t in tables:
        if not t:
            continue

        header = t[0]
        body_rows = t[1:]

        idxs = find_header_indexes(header, cfg)
        if not idxs:
            idxs = infer_indexes_from_table(body_rows)
            if DEBUG_LOG:
                ht = [txt for (txt, _h) in header]
                log.info(f"[diag] header={ht} -> inferred idxs={idxs}")

        if not idxs:
            if DEBUG_LOG:
                log.info("[diag] Could not determine date/presenter columns; skipping table.")
            continue

        idx_date, idx_presenter, _idx_topic = idxs
        if DEBUG_LOG:
            log.info(f"[diag] using columns -> date={idx_date}, presenter={idx_presenter}")

        for row in body_rows:
            if max(idx_date, idx_presenter) >= len(row):
                continue

            (date_text, date_html) = row[idx_date]
            (presenter_text, presenter_html) = row[idx_presenter]

            date_text = (date_text or "").strip()
            presenter_text = (presenter_text or "").strip()

            # Must be a mention (literal '@' or real macro) if require_at=True
            literal_at = presenter_text.startswith("@")
            mention_elem = is_confluence_user_mention(presenter_html)
            if cfg.require_at and not (literal_at or mention_elem):
                if DEBUG_LOG:
                    log.info(f"Skip: presenter not a mention -> '{presenter_text}'")
                continue

            # Parse date: HTML first (<time> / data-timestamp), else text
            dt = parse_confluence_date_from_html(date_html, cfg.timezone) or parse_date(date_text, cfg.timezone)
            if not dt:
                if DEBUG_LOG:
                    has_time = "<time" in (date_html or "")
                    has_node = 'data-node-type="date"' in (date_html or "")
                    log.info(f"Skip: unparseable date -> '{date_text}' / html has <time>={has_time}, date-node={has_node}")
                continue

            # Resolve display label (handles empty visible text for mentions)
            label = presenter_label_from_cell(presenter_text, presenter_html, cfg.require_at)
            if cfg.require_at and not label:
                if DEBUG_LOG:
                    log.info("Skip: mention present but no resolvable name (accountId lookup failed).")
                continue

            items.append((dt, label or presenter_text))

    # De-dup + sort
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
        # Stable across same ISO week number
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

    if not cfg.accept_multiple:
        entries = entries[:1]

    lines = []
    if len(entries) == 1:
        d, p = entries[0]
        p = slackify_presenter(p, cfg)
        date_str = d.strftime("%d %B %Y")  # e.g., 25 September 2025
        tmpl = pick_template(cfg, has_presenter=True, now_local=d)
        lines.append(tmpl.format(presenter=p, date=date_str))
    else:
        for d, p in entries:
            p = slackify_presenter(p, cfg)
            date_str = d.strftime("%d %B %Y")
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
                text = text or ""
                html = html or ""
                # Signals for debugging
                date_from_html = parse_confluence_date_from_html(html, cfg.timezone)
                date_from_text = parse_date(text, cfg.timezone)
                m = re.search(r'data-timestamp="(\d+)"', html, flags=re.I)
                ts_ms = int(m.group(1)) if m else None

                row_entry.append({
                    "col_index": ci,
                    "text": text.strip(),
                    "html_snippet": html[:600],  # trim to keep artifact manageable
                    "has_time_tag": ("<time" in html and 'datetime="' in html),
                    "has_date_node": ('data-node-type="date"' in html),
                    "date_ts_ms": ts_ms,
                    "parsed_date_iso_html": date_from_html.isoformat() if date_from_html else None,
                    "parsed_date_iso_text": date_from_text.isoformat() if date_from_text else None,
                    "mention_elem": is_confluence_user_mention(html),
                    "account_ids": extract_account_ids(html),
                    "literal_at": text.strip().startswith("@"),
                })
            table_entry["rows"].append(row_entry)
        payload.append(table_entry)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    log.info(f"[dump] wrote {out_path} (tables={len(payload)})")



# ----------------- Weekly 1:1 pairing -----------------
def _load_json(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    except FileNotFoundError:
        return {}
    except Exception as e:
        log.info(f"[pairing] Could not read state file '{path}': {e}")
        return {}

def _save_json(path: str, data: dict) -> None:
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        log.info(f"[pairing] Could not write state file '{path}': {e}")

def _iso_week_key(dt: datetime) -> str:
    y, w, _ = dt.isocalendar()
    return f"{y}-W{w:02d}"

def pick_weekly_pair(cfg: BotConfig, now_local: datetime) -> Optional[Tuple[str, str]]:
    """
    Pick one random pair per ISO week.
    - Uses everyone in cfg.pairing_user_ids exactly once per cycle before repeating.
    - If odd number of users, leftover is carried into next week and paired first.
    - Stable within the same ISO week (reruns won't change the pair).
    Returns mentions ("<@U...>", "<@U...>") or None.
    """
    if not cfg.pairing_enabled:
        return None

    eligible = [u for u in (cfg.pairing_user_ids or []) if u]
    eligible = list(dict.fromkeys(eligible))  # de-dupe
    if len(eligible) < 2:
        return None

    state = _load_json(cfg.pairing_state_path)
    week_key = _iso_week_key(now_local)

    # If already chosen for this week, reuse (stability across reruns)
    if state.get("week_key") == week_key and isinstance(state.get("pair"), list) and len(state["pair"]) == 2:
        a, b = state["pair"]
        if a and b:
            return (f"<@{a}>", f"<@{b}>")

    carry = state.get("carry")  # user id or None
    remaining = state.get("remaining") or []

    # Clean/validate against current eligible list
    if carry and carry not in eligible:
        carry = None
    remaining = [u for u in remaining if u in eligible and u != carry]

    # Start a fresh cycle if needed
    if not remaining:
        remaining = [u for u in eligible if u != carry]
        rnd = random.Random(week_key)  # deterministic shuffle seed for this week's draw
        rnd.shuffle(remaining)

    # Pick first user
    if carry:
        a_id = carry
        carry = None
    else:
        a_id = remaining.pop(0)

    # Ensure second user exists; if not, start a new cycle excluding a_id
    if not remaining:
        remaining = [u for u in eligible if u != a_id]
        rnd = random.Random(week_key + ":refill")
        rnd.shuffle(remaining)

    b_id = remaining.pop(0)

    # If exactly one user remains, carry them to next week
    carry_next = None
    if len(remaining) == 1:
        carry_next = remaining.pop(0)

    _save_json(cfg.pairing_state_path, {
        "week_key": week_key,
        "pair": [a_id, b_id],
        "remaining": remaining,
        "carry": carry_next,
        "updated_at": datetime.now(pytz.UTC).isoformat(),
    })

    return (f"<@{a_id}>", f"<@{b_id}>")



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

    # Force a specific date (useful for reproducing a known row)
    if FORCE_DATE_STR:
        forced = dtparse.parse(FORCE_DATE_STR)
        now = forced.astimezone(tz) if forced.tzinfo else tz.localize(forced)

    # Select time window
    if TEST_TODAY_ONLY or FORCE_DATE_STR:
        start, end = day_bounds(now, cfg.timezone)
        log.info(f"Today-only window: {start.date()} [{cfg.timezone}]")
    else:
        start, end = week_bounds(now, cfg.timezone)
        log.info(f"Week window: {start.date()} → {end.date()} [{cfg.timezone}]")

    # Fetch + parse Confluence page
    html = fetch_confluence_storage_html()
    tables = parse_tables(html)

    # Dump-only mode: write artifact & exit (no Slack)
    if PARSE_DUMP_ONLY:
        dump_full_tables(tables, cfg, out_path="/tmp/confluence_tables_dump.json")
        log.info("Dump-only mode: wrote /tmp/confluence_tables_dump.json and exited.")
        return

    # Light diagnostics
    if DEBUG_LOG:
        dump_parse_diagnostics(tables, cfg, limit=300, out_path="/tmp/presenter_parse_debug.json")

    # Build schedule (handles <time> + data-node dates and @mentions)
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

    # Format weekly presenter message
    message = format_message(entries, cfg)

    # Append weekly 1:1 pairing
    if cfg.pairing_enabled:
        pair = pick_weekly_pair(cfg, now_local=now)
        if pair:
            a, b = pair
            message += f"\n\n{cfg.pairing_header}\n" + cfg.pairing_line_template.format(a=a, b=b)

    # Send to Slack
    post_to_slack(message)
    log.info("Posted to Slack.")



if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log.error(str(e))
        sys.exit(1)
