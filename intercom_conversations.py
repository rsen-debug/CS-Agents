"""
CS Intercom Conversations Summarizer
--------------------------------------
Pulls recent Intercom conversations for a segment of ChurnZero accounts,
asks an LLM to summarize each into pain_point / action_taken / sentiment,
and writes the results back into a ChurnZero custom table.

Two fixes vs. the original notebook, required to run unattended:

1. ALL credentials come from environment variables (GitHub Actions secrets)
   instead of being hardcoded.
2. The original notebook saved the final CSV under one filename
   ("df_final_Aug_12.csv") but then tried to POST a *different* filename
   ("conversations_Intercom.csv") that was never actually written — that
   upload step would have silently failed (FileNotFoundError) every run.
   This version writes and posts the same file.
3. The 1-day Intercom lookback is now LOOKBACK_HOURS (default 24), since the
   job runs every 12 hours — a 24h window gives a buffer in case a run is
   ever late or fails.
"""

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
from tqdm.auto import tqdm


def require_env(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        print(f"ERROR: required environment variable {name} is not set.", file=sys.stderr)
        sys.exit(1)
    return val


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

# ChurnZero
CZ_BASE_URL = os.environ.get("CZ_BASE_URL", "https://cloudways.eu1app.churnzero.net")
CZ_AUTH_HEADER = require_env("CZ_AUTH_HEADER")
CZ_IMPORT_APP_KEY = require_env("CZ_IMPORT_APP_KEY")
CZ_CUSTOM_TABLE_ID = os.environ.get("CZ_CUSTOM_TABLE_ID", "29")
CZ_SEGMENT_FILTER = os.environ.get("CZ_SEGMENT_FILTER", "Use/SegmentId eq 3368")

CZ_HEADERS = {
    "Authorization": CZ_AUTH_HEADER,
    "Content-Type": "application/json",
}

# Intercom
INTERCOM_ACCESS_TOKEN = require_env("INTERCOM_ACCESS_TOKEN")
INTERCOM_BASE_URL = "https://api.intercom.io"
INTERCOM_HEADERS = {
    "Authorization": f"Bearer {INTERCOM_ACCESS_TOKEN}",
    "Content-Type": "application/json",
    "Accept": "application/json",
    "Intercom-Version": "2.16",
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

# Runs every 12h; look back 24h by default for safety margin
LOOKBACK_HOURS = int(os.environ.get("LOOKBACK_HOURS", "24"))
conversation_time_filter = int(time.time()) - (LOOKBACK_HOURS * 60 * 60)

OUTPUT_CSV = "conversations_Intercom.csv"

print(f"Looking back {LOOKBACK_HOURS}h (since epoch {conversation_time_filter}) for Intercom conversations")


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


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Pull Intercom conversations for each ChurnZero account (Primary User ID)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_conversations_for_users(primary_user_ids) -> list:
    all_full_conversations_details = []
    processed_conversation_ids = set()

    for current_primary_user_id in tqdm(primary_user_ids, desc="Processing Primary User IDs"):
        # --- Method 1: Direct conversation search by Primary User ID ---
        direct_search_conversations = []
        body_direct_conv_search = {
            "query": {
                "operator": "AND",
                "value": [
                    {"field": "custom_attributes.Primary user id", "operator": "=", "value": int(current_primary_user_id)},
                    {"field": "created_at", "operator": ">", "value": conversation_time_filter},
                ],
            },
            "pagination": {"per_page": 50},
        }
        while True:
            try:
                resp = requests.post(f"{INTERCOM_BASE_URL}/conversations/search", headers=INTERCOM_HEADERS, json=body_direct_conv_search)
                resp.raise_for_status()
                data = resp.json()
                direct_search_conversations.extend(data["conversations"])
                next_page = data.get("pages", {}).get("next")
                if not next_page:
                    break
                body_direct_conv_search["pagination"] = {"starting_after": next_page["starting_after"], "per_page": 50}
                time.sleep(0.2)
            except requests.exceptions.RequestException as e:
                print(f"Error direct searching conversations for Primary User ID {current_primary_user_id}: {e}")
                break

        # --- Method 2: Conversation search via contact IDs ---
        contact_ids_for_user = set()
        body_contact_search = {
            "query": {"field": "external_id", "operator": "=", "value": current_primary_user_id}
        }
        try:
            resp_contacts = requests.post(f"{INTERCOM_BASE_URL}/contacts/search", headers=INTERCOM_HEADERS, json=body_contact_search)
            resp_contacts.raise_for_status()
            contacts_data = resp_contacts.json()
            for contact_item in contacts_data.get("data", []):
                contact_ids_for_user.add(contact_item["id"])
        except requests.exceptions.RequestException as e:
            print(f"Error searching contacts for Primary User ID {current_primary_user_id}: {e}")
        time.sleep(0.2)

        contact_search_conversations = []
        for contact_id in contact_ids_for_user:
            body_contact_conv_search = {
                "query": {
                    "operator": "AND",
                    "value": [
                        {"field": "contact_ids", "operator": "=", "value": contact_id},
                        {"field": "updated_at", "operator": ">", "value": conversation_time_filter},
                    ],
                },
                "pagination": {"per_page": 50},
            }
            while True:
                try:
                    resp = requests.post(f"{INTERCOM_BASE_URL}/conversations/search", headers=INTERCOM_HEADERS, json=body_contact_conv_search)
                    resp.raise_for_status()
                    data = resp.json()
                    contact_search_conversations.extend(data["conversations"])
                    next_page = data.get("pages", {}).get("next")
                    if not next_page:
                        break
                    body_contact_conv_search["pagination"] = {"starting_after": next_page["starting_after"], "per_page": 50}
                    time.sleep(0.2)
                except requests.exceptions.RequestException as e:
                    print(f"Error searching conversations for contact {contact_id} (Primary User ID {current_primary_user_id}): {e}")
                    break

        # --- Combine and fetch full details for unique conversations ---
        all_conversations_for_user = direct_search_conversations + contact_search_conversations
        unique_conv_summaries = {c["id"]: c for c in all_conversations_for_user}

        for conv_id in unique_conv_summaries:
            if conv_id in processed_conversation_ids:
                continue
            try:
                detail_resp = requests.get(
                    f"{INTERCOM_BASE_URL}/conversations/{conv_id}",
                    headers=INTERCOM_HEADERS,
                    params={"display_as": "plaintext"},
                )
                detail_resp.raise_for_status()
                all_full_conversations_details.append((detail_resp.json(), str(current_primary_user_id)))
                processed_conversation_ids.add(conv_id)
            except requests.exceptions.RequestException as e:
                print(f"Failed to fetch full detail for conversation {conv_id} (Primary User ID {current_primary_user_id}): {e}")
            time.sleep(0.2)

    print(f"Total unique full conversations collected: {len(all_full_conversations_details)}")
    return all_full_conversations_details


def extract_conversation_text(conv) -> str:
    lines = []
    source = conv.get("source", {})
    author = source.get("author", {})
    author_type = author.get("type", "unknown")
    author_name = author.get("name", "")
    source_body = source.get("body", "")
    if source_body:
        lines.append(f"[{author_type}] {author_name}: {source_body}")

    parts = conv.get("conversation_parts", {}).get("conversation_parts", [])
    for part in parts:
        part_author = part.get("author", {})
        body = part.get("body")
        if body:
            part_author_type = part_author.get("type", "unknown")
            part_author_name = part_author.get("name", "")
            lines.append(f"[{part_author_type}] {part_author_name}: {body}")

    return "\n".join(lines)


def build_df_full(all_full_conversations_details) -> pd.DataFrame:
    rows = []
    for conv_data, user_id in all_full_conversations_details:
        conversation_rating_data = conv_data.get("conversation_rating")
        chat_rating = conversation_rating_data.get("rating") if conversation_rating_data else None
        rows.append(
            {
                "conversation_id": conv_data["id"],
                "created_at": conv_data.get("created_at"),
                "updated_at": conv_data.get("updated_at"),
                "content": extract_conversation_text(conv_data),
                "Primary_User_ID": user_id,
                "chat_rating": chat_rating,
            }
        )
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: LLM summarization
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a Customer Support QA analyst. "
    "Read the full support ticket/conversation below between a customer and a support agent. "
    "Summarize it into exactly three fields:\n"
    "1. pain_point: the customer's core issue or complaint, in one concise sentence.\n"
    "2. action_taken: what the support agent did or resolved, in one concise sentence. "
    "3. sentiment: the customer's overall sentiment, if the customer is repeatedly expressing appreciation of the "
    "support agent's effort, mark it as positive — must be exactly one of: "
    "\"Positive\", \"Neutral\", or \"Frustrated\".\n\n"
    "Return ONLY a valid JSON object with these three keys, no other text, no markdown formatting."
)


def summarize_ticket(conv_id, context: str):
    empty_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    if not context.strip():
        return conv_id, {"pain_point": "No content available.", "action_taken": "", "sentiment": "Neutral"}, empty_usage
    try:
        resp = requests.post(
            AGENT_URL + "/chat/completions",
            headers=HEADERS,
            json={
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Conversation ID: {conv_id}\n\n=== Ticket Content ===\n{context}"},
                ],
                "temperature": 0.2,
                "top_p": 1,
                "max_tokens": 300,
            },
            timeout=60,
        )
        resp.raise_for_status()
        body = resp.json()
        raw = body["choices"][0]["message"]["content"].strip()
        usage = body.get("usage", empty_usage)
        if raw.startswith("```"):
            raw = raw.strip("`")
            raw = raw.replace("json\n", "", 1).strip()
        parsed = json.loads(raw)
        return conv_id, parsed, usage
    except Exception as e:
        return conv_id, {"pain_point": f"[Error: {e}]", "action_taken": "", "sentiment": "Neutral"}, empty_usage


# ─────────────────────────────────────────────────────────────────────────────
# Step 4: Push results back into ChurnZero custom table
# ─────────────────────────────────────────────────────────────────────────────

def push_to_churnzero(csv_path: str):
    url = f"{CZ_BASE_URL}/batchCustomTableCsv?customTableId={CZ_CUSTOM_TABLE_ID}&appKey={CZ_IMPORT_APP_KEY}"
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

    primary_user_ids = df["ExternalId"].tolist()

    conversations = fetch_conversations_for_users(primary_user_ids)
    if not conversations:
        print("No Intercom conversations found in lookback window. Exiting.")
        return

    df_full = build_df_full(conversations)
    print(f"Built df_full with shape {df_full.shape}")

    ticket_contexts = dict(
        zip(df_full["conversation_id"], df_full["content"].fillna("").str.slice(0, MAX_INPUT_CHARS))
    )
    print(f"Built context for {len(ticket_contexts):,} tickets")

    results = {}
    total_prompt_tokens = 0
    total_completion_tokens = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(summarize_ticket, conv_id, ctx): conv_id for conv_id, ctx in ticket_contexts.items()}
        with tqdm(total=len(futures), desc="Summarizing tickets") as pbar:
            for future in as_completed(futures):
                conv_id, summary, usage = future.result()
                results[conv_id] = summary
                total_prompt_tokens += usage.get("prompt_tokens", 0)
                total_completion_tokens += usage.get("completion_tokens", 0)
                pbar.update(1)

    total_tokens = total_prompt_tokens + total_completion_tokens
    n_tickets = max(len(results), 1)
    print(f"Done — summaries generated for {len(results):,} tickets.")
    print(
        f"Token usage this run: {total_tokens:,} total "
        f"({total_prompt_tokens:,} input / {total_completion_tokens:,} output) "
        f"across {len(results):,} tickets — avg {total_tokens / n_tickets:,.0f} tokens/ticket"
    )

    log_path = "token_usage_log.csv"
    log_row = pd.DataFrame([{
        "run_timestamp_utc": pd.Timestamp.utcnow().isoformat(),
        "n_tickets": len(results),
        "prompt_tokens": total_prompt_tokens,
        "completion_tokens": total_completion_tokens,
        "total_tokens": total_tokens,
        "avg_tokens_per_ticket": round(total_tokens / n_tickets, 1),
    }])
    log_row.to_csv(log_path, mode="a", header=not os.path.exists(log_path), index=False)

    df_summaries = pd.DataFrame(
        [
            {
                "conversation_id": conv_id,
                "pain_point": summary.get("pain_point", ""),
                "action_taken": summary.get("action_taken", ""),
                "sentiment": summary.get("sentiment", ""),
            }
            for conv_id, summary in results.items()
        ]
    )

    df_final = df_full.merge(df_summaries, on="conversation_id", how="left")
    print(f"Result shape: {df_final.shape}")
    print("Sentiment distribution:")
    print(df_final["sentiment"].value_counts())

    df_final = df_final.rename(
        columns={
            "conversation_id": "ConversationId",
            "created_at": "Createdat",
            "updated_at": "Updatedat",
            "content": "Content",
            "pain_point": "PainPoint",
            "action_taken": "ActionTaken",
            "sentiment": "Sentiment",
            "Primary_User_ID": "UId",
            "chat_rating": "Rating",
        }
    )
    df_final["Createdat"] = pd.to_datetime(df_final["Createdat"], unit="s")
    df_final["Updatedat"] = pd.to_datetime(df_final["Updatedat"], unit="s")

    df_final.to_csv(OUTPUT_CSV, index=False)
    print(f"Wrote {len(df_final)} rows to {OUTPUT_CSV}")
    print(df_final.head())

    push_to_churnzero(OUTPUT_CSV)


if __name__ == "__main__":
    main()
