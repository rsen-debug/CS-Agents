"""
CS Top-50 Email Recommendation Engine
--------------------------------------
Reads recent Close.io activity (emails/calls/meetings/notes) for a segment of
ChurnZero accounts, asks an LLM for a recommended next action per customer,
and writes the recommendations back into ChurnZero.

This is a straight port of the original notebook, with two changes required
to run unattended on a schedule:

1. ALL credentials now come from environment variables (set as GitHub Actions
   secrets) instead of being hardcoded in the file.
2. The "date_created__gte" lookback is now computed relative to today instead
   of a hardcoded date, using LOOKBACK_DAYS (default 3, i.e. schedule interval
   of 2 days + 1 day buffer so nothing slips through if a run is late/fails).
"""

import ast
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta

import pandas as pd
import requests
from bs4 import BeautifulSoup
from requests.auth import HTTPBasicAuth

# ─────────────────────────────────────────────────────────────────────────────
# Config — all secrets/env-specific values pulled from the environment
# ─────────────────────────────────────────────────────────────────────────────

def require_env(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        print(f"ERROR: required environment variable {name} is not set.", file=sys.stderr)
        sys.exit(1)
    return val

# ChurnZero
CZ_BASE_URL = os.environ.get("CZ_BASE_URL", "https://cloudways.eu1app.churnzero.net")
CZ_AUTH_HEADER = require_env("CZ_AUTH_HEADER")          # full "Basic xxxxx..." string
CZ_IMPORT_APP_KEY = require_env("CZ_IMPORT_APP_KEY")    # appKey for batchAccountsCsv
CZ_SEGMENT_FILTER = os.environ.get(
    "CZ_SEGMENT_FILTER",
    "Use/SegmentId eq 3351 and Use/SegmentColumnSetId eq 197",
)

CZ_HEADERS = {
    "Authorization": CZ_AUTH_HEADER,
    "Content-Type": "application/json",
}

# Close.io
CLOSE_API_KEY = require_env("CLOSE_API_KEY")
CLOSE_AUTH = HTTPBasicAuth(CLOSE_API_KEY, "")
CLOSE_CUSTOM_FIELD_ID = os.environ.get(
    "CLOSE_CUSTOM_FIELD_ID", "lcf_TzEz3UaB7Dq278IRtwIUhKFCSmfmtXPnbA3X5dCB4iy"
)
CLOSE_HEADERS = {
    "Content-Type": "application/json",
    "Referer": "https://app.close.com/",
}

# DigitalOcean Serverless Inference (LLM)
AGENT_URL = os.environ.get("DO_AGENT_URL", "https://inference.do-ai.run/v1")
DO_API_KEY = require_env("DO_API_KEY")
HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {DO_API_KEY}",
}
MODEL = os.environ.get("DO_MODEL", "anthropic-claude-4.5-sonnet")
MAX_INPUT_CHARS = 6000
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", "5"))

# Lookback window: schedule runs every 2 days, default to 3 days for safety margin
LOOKBACK_DAYS = int(os.environ.get("LOOKBACK_DAYS", "3"))
today_date = date.today()
lookback_date = (today_date - timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d")

print(f"Run date: {today_date} | Fetching activity since: {lookback_date}")

# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Pull the ChurnZero account segment
# ─────────────────────────────────────────────────────────────────────────────

def fetch_churnzero_accounts() -> pd.DataFrame:
    endpoint = f"{CZ_BASE_URL}/public/v1/Account"
    params = {"$filter": CZ_SEGMENT_FILTER}

    all_records = []
    url = endpoint
    while url:
        if url == endpoint:
            resp = requests.get(url, headers=CZ_HEADERS, params=params, timeout=30)
        else:
            resp = requests.get(url, headers=CZ_HEADERS, timeout=30)

        if resp.status_code != 200:
            print(f"Error {resp.status_code}: {resp.text[:300]}")
            resp.raise_for_status()

        payload = resp.json()
        all_records.extend(payload.get("value", []))
        url = payload.get("@odata.nextLink")

    if not all_records:
        print("No accounts found for configured segment.")
        return pd.DataFrame()

    df = pd.DataFrame(all_records)
    for col in ["Createdat", "Updatedat", "SolvedAt"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    print(f"Fetched {len(df)} accounts")
    return df


def json_loads_safe(s):
    return json.loads(s)


def _parse_cf(cf):
    if isinstance(cf, dict):
        return cf
    if isinstance(cf, str) and cf.strip():
        for parser in (json_loads_safe, ast.literal_eval):
            try:
                parsed = parser(cf)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass
    return {}


def get_support_scope(cf):
    cf = _parse_cf(cf)
    for key, value in cf.items():
        k = key.lower().replace("_", " ")
        if "support" in k and "scope" in k:
            return value
    return None


def is_premium(scope) -> bool:
    return isinstance(scope, str) and "premium" in scope.lower()


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Resolve Close.io lead IDs for each ChurnZero account (CW ID)
# ─────────────────────────────────────────────────────────────────────────────

def _parse_rate_limit_wait(headers, sep=";") -> float:
    rate_limit_header = headers.get("RateLimit", "")
    for part in rate_limit_header.split(sep):
        if "reset" in part:
            try:
                return float(part.split("=")[1].strip())
            except (ValueError, IndexError):
                pass
    return 10.0


def fetch_lead_ids(cw_ids) -> pd.DataFrame:
    url_search = "https://api.close.com/api/v1/data/search/"
    url_lead_detail = "https://api.close.com/api/v1/lead/"
    all_leads_with_cw_id_data = []

    for single_cw_id in cw_ids:
        while True:
            payload = {
                "query": {
                    "type": "and",
                    "queries": [
                        {"type": "object_type", "object_type": "lead"},
                        {
                            "type": "field_condition",
                            "field": {"type": "custom_field", "custom_field_id": CLOSE_CUSTOM_FIELD_ID},
                            "condition": {"type": "text", "value": single_cw_id, "mode": "phrase"},
                        },
                    ],
                }
            }
            resp = requests.post(url_search, headers=CLOSE_HEADERS, json=payload, auth=CLOSE_AUTH)
            if resp.status_code == 429:
                wait_time = _parse_rate_limit_wait(resp.headers, ";")
                print(f"Rate limited on search for CW ID {single_cw_id}. Waiting {wait_time}s...")
                time.sleep(wait_time)
                continue
            break

        print(f"Search API status code for CW ID {single_cw_id}: {resp.status_code}")

        if resp.status_code == 200:
            search_results = resp.json()["data"]
            lead_ids_from_search = [item["id"] for item in search_results]

            for lead_id in lead_ids_from_search:
                while True:
                    params_lead_detail = {"_fields": f"id,custom.{CLOSE_CUSTOM_FIELD_ID}"}
                    r = requests.get(
                        f"{url_lead_detail}{lead_id}/", params=params_lead_detail, auth=CLOSE_AUTH
                    )
                    if r.status_code == 429:
                        wait_time = _parse_rate_limit_wait(r.headers, ";")
                        print(f"Rate limited on lead detail for {lead_id}. Waiting {wait_time}s...")
                        time.sleep(wait_time)
                        continue
                    break
                r.raise_for_status()
                lead_data = r.json()
                all_leads_with_cw_id_data.append(
                    {"lead_id": lead_data["id"], "CW ID": lead_data.get(f"custom.{CLOSE_CUSTOM_FIELD_ID}")}
                )
        else:
            print(f"Search API request failed for CW ID {single_cw_id} with status {resp.status_code}")
            try:
                print("Error response:", resp.json())
            except json.JSONDecodeError:
                print("Error response (non-JSON):", resp.text)

    return pd.DataFrame(all_leads_with_cw_id_data)


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Pull emails / meetings / calls / notes for those leads since lookback_date
# ─────────────────────────────────────────────────────────────────────────────

def fetch_activity(kind: str, lead_ids, df_lead_cw_id: pd.DataFrame) -> pd.DataFrame:
    """kind is one of: email, meeting, call, note"""
    url = f"https://api.close.com/api/v1/activity/{kind}/"
    all_items = []
    limit = 100

    for single_lead_id in lead_ids:
        skip = 0
        while True:
            params = {
                "_limit": limit,
                "_skip": skip,
                "date_created__gte": lookback_date,
                "lead_id": single_lead_id,
            }
            while True:
                resp = requests.get(url, params=params, auth=CLOSE_AUTH)
                if resp.status_code == 429:
                    wait_time = _parse_rate_limit_wait(resp.headers, ",")
                    print(f"Rate limited fetching {kind} for lead {single_lead_id}. Waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                break
            resp.raise_for_status()
            data = resp.json()
            all_items.extend(data["data"])
            if not data["has_more"]:
                break
            skip += limit

    df_raw = pd.DataFrame(all_items)
    print(f"Fetched {len(df_raw)} {kind} rows")
    return df_raw


def build_df_emails(df_emails_raw: pd.DataFrame, df_lead_cw_id: pd.DataFrame) -> pd.DataFrame:
    if df_emails_raw.empty:
        return pd.DataFrame(
            columns=[
                "CW ID", "lead_id", "user_name", "date_created", "direction",
                "subject", "body_html", "created_by", "sender", "thread_id", "status",
            ]
        )
    df_merged = pd.merge(df_emails_raw, df_lead_cw_id, on="lead_id", how="left")
    cols = [
        "CW ID", "lead_id", "user_name", "date_created", "direction",
        "subject", "body_html", "created_by", "sender", "thread_id", "status",
    ]
    df_merged = df_merged[[c for c in cols if c in df_merged.columns]]
    df_merged["date_created"] = pd.to_datetime(df_merged["date_created"], format="ISO8601")
    df_merged["activity_type"] = "email"
    df_merged = df_merged[df_merged["sender"] != "Team Cloudways <success@cloudways.com>"]
    return df_merged


def build_generic_activity(df_raw: pd.DataFrame, df_lead_cw_id: pd.DataFrame, kind: str, cols, date_cols) -> pd.DataFrame:
    if df_raw.empty:
        out = pd.DataFrame(columns=cols)
        out["activity_type"] = kind
        return out
    for col in date_cols:
        if col in df_raw.columns:
            df_raw[col] = df_raw[col].apply(lambda x: str(x) if x is not None and not isinstance(x, (str, int, float)) else x)
            df_raw[col] = pd.to_datetime(df_raw[col], errors="coerce", format="ISO8601")
    df_merged = pd.merge(df_raw, df_lead_cw_id, on="lead_id", how="left")
    df_merged = df_merged[[c for c in cols if c in df_merged.columns]]
    df_merged["activity_type"] = kind
    return df_merged


# ─────────────────────────────────────────────────────────────────────────────
# Step 4: LLM recommendation generation
# ─────────────────────────────────────────────────────────────────────────────

def html_to_text(html):
    if not isinstance(html, str) or not html.strip():
        return ""
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(separator=" ", strip=True)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_customer_context(group: pd.DataFrame) -> str:
    parts = []
    for _, row in group.sort_values("date_created").iterrows():
        direction = row.get("direction", "")
        direction_label = "Customer" if direction == "incoming" else "Agent"
        subject = row.get("subject", "") or ""
        body = row.get("body_text", "") or ""
        d = str(row.get("date_created", ""))[:10]
        snippet = f"[{d}] {direction_label}: {subject}\n{body[:800]}"
        parts.append(snippet)
    full_text = "\n\n---\n\n".join(parts)
    return full_text[:MAX_INPUT_CHARS]


SYSTEM_PROMPT = (
    f"You are a Customer Success analyst and today is {today_date}. "
    "Look at the last email between the customer and a support/CS team, ignore login notifications, "
    "Context from emails from CSMs get priority over emails from success@cloudways.com"
    "identify the most important next course of action for the Customer Success Manager. "
    "Return EXACTLY 1 concise, actionable recommendation with deadline within 50 words max. "
    "Focus on retention, expansion, issue resolution, and relationship health. "
    "Be specific and practical — avoid generic advice. "
    "You will be told the customer's support scope. "
    "Do not assume that the customer needs to be off-boarded or their business has closed down."
    "If the support scope is Premium, explicitly instruct the CSM to carry out the "
    "recommended reach-out via the customer's dedicated Slack premium channel "
    "(not email). For all other support scopes, do not mention Slack."
)


def get_recommendations(cw_id, context: str, support_scope=None):
    if not context.strip():
        return cw_id, "No email content available."

    scope_label = support_scope if support_scope else "Unknown"
    if is_premium(support_scope):
        scope_instruction = (
            "This customer's support scope is PREMIUM. Your recommendation MUST "
            "instruct the CSM to carry out the reach-out via the customer's "
            "dedicated Slack premium channel."
        )
    else:
        scope_instruction = (
            f"This customer's support scope is: {scope_label}. "
            "Recommend the standard reach-out channel (e.g. email/call)."
        )

    try:
        resp = requests.post(
            AGENT_URL + "/chat/completions",
            headers=HEADERS,
            json={
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"Customer CW ID: {cw_id}\n"
                            f"Support Scope: {scope_label}\n"
                            f"{scope_instruction}\n\n"
                            "=== Email History ===\n"
                            + context
                            + "\n\n"
                            "Based on the above email history, what are the 2-3 most important "
                            "next actions the Customer Success Manager should take for this customer?"
                        ),
                    },
                ],
                "temperature": 0.2,
                "top_p": 1,
                "max_tokens": 512,
            },
            timeout=60,
        )
        resp.raise_for_status()
        return cw_id, resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return cw_id, f"[Error: {e}]"


# ─────────────────────────────────────────────────────────────────────────────
# Step 5: Push recommendations back into ChurnZero
# ─────────────────────────────────────────────────────────────────────────────

def push_to_churnzero(csv_path: str):
    url = f"{CZ_BASE_URL}/batchAccountsCsv?appKey={CZ_IMPORT_APP_KEY}"
    with open(csv_path, "rb") as f:
        files = {"file": (csv_path, f, "text/csv")}
        resp = requests.post(url, files=files)
        print(f"Status Code: {resp.status_code}")
        print(f"Response: {resp.text}")
        resp.raise_for_status()
        print("File successfully posted!")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    df = fetch_churnzero_accounts()
    if df.empty or "ExternalId" not in df.columns:
        print("No accounts to process. Exiting.")
        return

    cw_ids = df["ExternalId"]
    df["support_scope"] = df["Cf"].apply(get_support_scope) if "Cf" in df.columns else None
    support_scope_map = dict(zip(df["ExternalId"], df["support_scope"]))

    print("Support scope distribution:")
    print(df["support_scope"].value_counts(dropna=False))
    print(f"Premium customers: {sum(is_premium(v) for v in support_scope_map.values())} of {len(support_scope_map)}")

    df_lead_cw_id = fetch_lead_ids(cw_ids)
    if df_lead_cw_id.empty:
        print("No Close.io leads matched. Exiting.")
        return
    lead_ids = df_lead_cw_id["lead_id"].tolist()

    df_emails_raw = fetch_activity("email", lead_ids, df_lead_cw_id)
    df_emails = build_df_emails(df_emails_raw, df_lead_cw_id)

    df_meetings_raw = fetch_activity("meeting", lead_ids, df_lead_cw_id)
    df_meetings = build_generic_activity(
        df_meetings_raw, df_lead_cw_id, "meeting",
        ["CW ID", "lead_id", "user_name", "date_created", "title", "starts_at", "created_by", "user_note", "attendees", "status"],
        ["starts_at", "activity_at", "date_created", "date_updated"],
    )

    df_calls_raw = fetch_activity("call", lead_ids, df_lead_cw_id)
    df_calls = build_generic_activity(
        df_calls_raw, df_lead_cw_id, "call",
        ["CW ID", "lead_id", "user_name", "date_created", "date_answered", "created_by", "note", "status", "duration"],
        ["date_answered", "activity_at", "date_created", "date_updated"],
    )

    df_notes_raw = fetch_activity("note", lead_ids, df_lead_cw_id)
    df_notes = build_generic_activity(
        df_notes_raw, df_lead_cw_id, "note",
        ["CW ID", "lead_id", "user_name", "date_created", "created_by", "note"],
        ["activity_at", "date_created", "date_updated"],
    )

    df_comm = pd.concat([df_calls, df_meetings, df_emails, df_notes], ignore_index=True)
    df_comm.sort_values(by="date_created", inplace=True)
    df_comm.reset_index(drop=True, inplace=True)

    if df_comm.empty or "body_html" not in df_comm.columns:
        print("No communication data found in lookback window. Exiting without recommendations.")
        return

    df_email_only = df_comm[df_comm["body_html"].notna() & (df_comm["body_html"].str.strip() != "")].copy()
    df_email_only["body_text"] = df_email_only["body_html"].apply(html_to_text)
    df_email_only = df_email_only[df_email_only["body_text"].str.strip() != ""]

    print(f"{len(df_email_only):,} email rows across {df_email_only['CW ID'].nunique():,} CW IDs")

    customer_contexts = (
        df_email_only.groupby("CW ID").apply(build_customer_context, include_groups=False).to_dict()
    )
    print(f"Built context for {len(customer_contexts):,} customers")

    results = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(get_recommendations, cw_id, ctx, support_scope_map.get(cw_id)): cw_id
            for cw_id, ctx in customer_contexts.items()
        }
        for future in as_completed(futures):
            cw_id, recs = future.result()
            results[cw_id] = recs

    print(f"Done — recommendations generated for {len(results):,} customers.")

    df_recommendations = (
        pd.DataFrame(results.items(), columns=["CW ID", "Recommendations"]).sort_values("CW ID").reset_index(drop=True)
    )
    df_recommendations = df_recommendations.rename(
        columns={"CW ID": "AccountExternalId", "Recommendations": "cf_RecommendedActionBasedOnCloseEmails"}
    )
    df_recommendations = df_recommendations[["AccountExternalId", "cf_RecommendedActionBasedOnCloseEmails"]]

    csv_path = "recomendations.csv"
    df_recommendations.to_csv(csv_path, index=False)
    print(f"Wrote {len(df_recommendations)} recommendations to {csv_path}")

    push_to_churnzero(csv_path)


if __name__ == "__main__":
    main()
