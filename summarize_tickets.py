import os
import re
import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.auto import tqdm

# ── DigitalOcean inference config ──────────────────────────────────────────
AGENT_URL = "https://inference.do-ai.run/v1"
HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {os.environ['DO_INFERENCE_TOKEN']}",
}
MODEL           = "openai-gpt-4.1"
MAX_INPUT_CHARS = 3000
MAX_WORKERS     = 5

# ── ChurnZero API config ───────────────────────────────────────────────────
CZ_BASE_URL = "https://cloudways.eu1app.churnzero.net"
CZ_HEADERS  = {
    "Authorization": f"Basic {os.environ['CZ_AUTH_TOKEN']}",
    "Content-Type": "application/json",
}
CZ_APP_KEY = os.environ["CZ_APP_KEY"]

print("Config ready...")


# ── ChurnZero: fetch Zendesk tickets ───────────────────────────────────────
def fetch_churnzero_tickets(top: int = 100) -> pd.DataFrame:
    endpoint = f"{CZ_BASE_URL}/public/v1/CustomListZendeskTickets"
    params = {
        "$filter": "Use/SegmentId eq 3194 and Use/SegmentColumnSetId eq 189",
        "$top": top,
    }

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
        print("No tickets found for segment 3194")
        return pd.DataFrame()

    df = pd.DataFrame(all_records)
    for col in ["Createdat", "Updatedat", "SolvedAt"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    print(f"Fetched {len(df)} tickets")
    return df


# ── Text utilities ─────────────────────────────────────────────────────────
def clean_text(text):
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# ── Summariser ─────────────────────────────────────────────────────────────
def summarise_one(idx, text):
    if not isinstance(text, str) or text.strip() == "":
        return idx, "No description provided."
    try:
        cleaned = clean_text(text)[:MAX_INPUT_CHARS]
        r = requests.post(
            AGENT_URL + "/chat/completions",
            headers=HEADERS,
            json={
                "model": MODEL,
                "messages": [{
                    "role": "user",
                    "content": (
                        "Summarise this support ticket in 2-3 sentences. "
                        "Write in third person (e.g. 'Customer reported...', 'Support team...'). "
                        "State the problem and what was done about it. Be concise and factual.\n\n"
                        "End with exactly: [Sentiment: Positive] or [Sentiment: Neutral] "
                        "or [Sentiment: Frustrated] or [Sentiment: Urgent] or [Sentiment: Angry]\n\n"
                        + cleaned
                    )
                }],
                "temperature": 0.3,
                "top_p": 1,
                "max_tokens": 512,
            },
            timeout=60,
        )
        r.raise_for_status()
        return idx, r.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return idx, f"[Error: {e}]"


def summarise_all(series, max_workers=MAX_WORKERS):
    results = {}
    items = list(series.items())
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(summarise_one, idx, text): idx for idx, text in items}
        with tqdm(total=len(items), desc="Summarising tickets") as pbar:
            for future in as_completed(futures):
                idx, summary = future.result()
                results[idx] = summary
                pbar.update(1)
    return results


print("Summariser ready.....")


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    raw_tickets_df = fetch_churnzero_tickets()

    if raw_tickets_df.empty:
        print("No tickets to summarize. Exiting.")
        return

    id_col = "TicketId" if "TicketId" in raw_tickets_df.columns else "Id"
    descriptions_to_summarise = raw_tickets_df.set_index(id_col)["Description"]

    summaries = summarise_all(descriptions_to_summarise)

    for ticket_id, summary in summaries.items():
        print(f"Ticket no: {ticket_id}")
        print(f"Summary:   {summary}")
        print("__________")

    if not summaries:
        print("No summaries generated. Exiting.")
        return

    df_summaries = pd.DataFrame(summaries.items(), columns=["Ticket ID", "Summary"])
    df_summaries["Ticket ID"] = df_summaries["Ticket ID"].astype(str)

    if "Account" not in raw_tickets_df.columns:
        print("Warning: 'Account' column not found. Cannot aggregate by AccountExternalId.")
        return

    temp_raw = raw_tickets_df[[id_col, "Account"]].copy()
    temp_raw["AccountExternalId"] = temp_raw["Account"].apply(
        lambda x: x.get("ExternalId") if isinstance(x, dict) else None
    )
    temp_raw[id_col] = temp_raw[id_col].astype(str)

    df_merged = pd.merge(
        df_summaries,
        temp_raw[[id_col, "AccountExternalId"]],
        left_on="Ticket ID",
        right_on=id_col,
        how="left",
    )

    df = (
        df_merged.groupby("AccountExternalId")
        .apply(lambda x: "\n\n".join(
            x.apply(
                lambda row: f"Ticket ID: {row['Ticket ID']}\nSummary: {row['Summary']}\n__________",
                axis=1,
            )
        ))
        .reset_index(name="cf_Last7DaysTicketsSummary")
    )

    print("\n--- Aggregated Ticket Summaries by AccountExternalId ---")
    print(df.head())

    # ── Save CSV ───────────────────────────────────────────────────────────
    csv_path = "sentiment_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved {csv_path}")

    # ── Push to ChurnZero ──────────────────────────────────────────────────
    url = f"{CZ_BASE_URL}/batchAccountsCsv?appKey={CZ_APP_KEY}"
    with open(csv_path, "rb") as f:
        files = {"file": (csv_path, f, "text/csv")}
        try:
            response = requests.post(url, files=files)
            print(f"Status Code: {response.status_code}")
            print(f"Response: {response.text}")
            response.raise_for_status()
            print("File successfully posted to ChurnZero!")
        except requests.exceptions.RequestException as e:
            print(f"An error occurred posting to ChurnZero: {e}")
            raise  # Re-raise so GitHub Actions marks the run as failed


if __name__ == "__main__":
    main()
