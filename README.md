# Weekly Prez Bot 🤖

The GitHub Actions bot reads a Confluence table and posts a weekly reminder in PUMA Slack channel about who is presenting that week.

---

## What it does

* Runs automatically **every Monday** at **08:00 UTC** (~09:00/10:00 Paris depending on DST).
* Reads a Confluence page (which has a table with columns: **Date**, **Presenter**, **Topic**).
* Picks the row whose **Date** falls in the current week (Mon–Sun). In testing mode it can target **today only**.
* Posts a Slack message using a **title**, a **blank line**, then a **templated body**. Templates can **rotate** weekly or be picked **randomly**.

If there’s no matching row for the week, it posts a message using one of the "no presenter" templates.

---

## Setup

### 1) Configure repository **secrets**

Add these in **Settings → Secrets and variables → Actions**:

Required:

* `CONFLUENCE_BASE_URL` – should be `https://criteo.atlassian.net/wiki`
* `CONFLUENCE_PAGE_ID` – numeric page ID from the Confluence URL
* `CONFLUENCE_USER` – Atlassian email that was used to create to the bot
* `CONFLUENCE_API_TOKEN` – Atlassian API token (**expires after ~1 year** in our setup; see *Maintenance* below)
* `SLACK_BOT_TOKEN` – Slack **Bot** token (app must have `chat:write` permission)
* `SLACK_CHANNEL_ID` – the channel to post to 

> **How to find IDs**
>
> * **Channel ID**: In Slack → open the channel → `View channel details` → About → Channel ID, or copy the channel link and take the `C…` part.
> * **User ID**: Open a user’s profile → `More` (⋯) → **Copy member ID** (`U…`).

### 2) Edit `config.yaml`

Key fields:

* `timezone`: usually `Europe/Paris`.
* `columns`: header names in your Confluence table (case-insensitive).
* `title`: the prefix line for all messages.
* `presenter_templates` / `no_presenter_templates`: the messages. You can use `\n` for line breaks or YAML block scalars (`|`).
* `template_mode`: `rotate`, `random`, or `random_per_run`.
* `require_presenter_at_mention`: if `true`, only rows with a real mention or `@Name` are accepted.
* `accept_multiple_in_week`: if `true`, will list all matching rows; otherwise only the first.
* `slack_user_map`: **IMPORTANT** – maps *normalized full name* to **Slack user ID** to render clickable mentions (see above how to obtain user ID).

#### Adding new team members (Slack IDs)

When someone joins the team:

1. Get their Slack **member ID** (`U…`), see above.
2. Add an entry under `slack_user_map` in `config.yaml`:

   ```yaml
   slack_user_map:
     "first last": "U0ABCDEF12"
   ```

> If a mapping is missing, the bot will post `@Name` as plain text (not blue/clickable and therefore won't ping that person).

### 3) Confluence source page

* `CONFLUENCE_PAGE_ID` comes from your Confluence URL, e.g.:

  * `https://…/wiki/spaces/SPACE/pages/1795892462/Thursday+Prez+2.0` → page ID `1795892462`.

---

## Scheduling

The workflow is scheduled via cron in `.github/workflows/weekly-presenter.yml`:

```yaml
schedule:
  - cron: "00 8 * * 1"   # Mondays 08:00 UTC
```

Paris is UTC+1 (winter) / UTC+2 (summer), so the message lands around 09:00/10:00 Paris local.

To tweak the time, adjust the cron. For example, **09:30 Paris** ≈ `07:30 UTC`:

```yaml
- cron: "30 7 * * 1"
```

---

## Testing (manual runs)

Use **Actions → Weekly Prez Bot → Run workflow**. Inputs:

* `PARSE_DUMP_ONLY` (default `false`): if `true`, write a full-page JSON dump to `/tmp/confluence_tables_dump.json` and exit (no Slack). The workflow uploads it as an artifact.
* `TEST_TODAY_ONLY` (default `false`): only consider rows matching **today** instead of the whole week.
* `FORCE_DATE` (optional): pretend today is this date, e.g. `2025-09-25` (useful to target a specific row).
* `DEBUG_LOG` (default `false`): more logs + a smaller `/tmp/presenter_parse_debug.json` artifact.

Artifacts appear on the run page under **Artifacts → parse-artifacts**.

---

## Maintenance

* **Confluence API token (IMPORTANT)**

  * Our Confluence token policy expires tokens **after ~1 year**. When it expires, updates will fail.
  * To refresh: create a new token in Atlassian → update the repo secret **`CONFLUENCE_API_TOKEN`** in **Settings → Secrets and variables → Actions**.
  * No code changes required.

* **Team changes**

  * For each new member, add their Slack ID to `slack_user_map` in `config.yaml` (see above).

* **Slack app reinstall / token rotation**

  * If you reinstall the Slack app or rotate credentials, update **`SLACK_BOT_TOKEN`** (and re-invite the bot to the channel if needed).

