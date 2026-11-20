"""
ChurnZero Onboarding Metrics
-----------------------------
Pipeline:
1. Fetch accounts (segment 3517) from ChurnZero
2. Fetch "Starting MRR" notes
3. Merge and compute cf_TimeToPickupinDays, cf_TimeToQualifyinDays, cf_MrrGrowth
4. Export df_final_onb.csv and push it back to ChurnZero

Changes vs. the original notebook, needed to run unattended:

1. Credentials come from environment variables (GitHub Actions secrets).
   CZ_AUTH_HEADER is the full "Basic ..." string, same as the other jobs.
2. A failed upload now fails the job. The notebook caught the exception and
   only printed it, so a scheduled run would show green with nothing uploaded.
3. display() calls and the hardcoded spot-check on account 2223996 removed
   (notebook/debug only).
4. Defensive handling for empty note results and for date columns that are
   missing from the custom fields, which would otherwise raise KeyError.
"""

import os
import re
import sys

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup


def require_env(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        print(f"ERROR: required environment variable {name} is not set.", file=sys.stderr)
        sys.exit(1)
    return val


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

# `or` (not a get-default) so an empty string from the workflow also falls back
CZ_BASE_URL = os.environ.get("CZ_BASE_URL") or "https://cloudways.eu1app.churnzero.net"
CZ_AUTH_HEADER = require_env("CZ_AUTH_HEADER")
CZ_IMPORT_APP_KEY = require_env("CZ_IMPORT_APP_KEY")
CZ_SEGMENT_FILTER = os.environ.get("CZ_SEGMENT_FILTER") or "Use/SegmentId eq 3517 and Use/SegmentColumnSetId eq 150"

CZ_HEADERS = {
    "Authorization": CZ_AUTH_HEADER,
    "Content-Type": "application/json",
}

OUTPUT_CSV = "df_final_onb.csv"

DATE_COLUMNS = [
    "CustomerPickedUpByOeDate",
    "CustomerQualifiedByOeDate",
    "CustomerUnqualifiedByOeDate",
]


def cz_get_all(entity: str, params: dict | None = None) -> list:
    """GET /public/v1/<entity> and follow @odata.nextLink pagination."""
    endpoint = f"{CZ_BASE_URL}/public/v1/{entity}"
    records, url, first = [], endpoint, True
    while url:
        resp = requests.get(url, headers=CZ_HEADERS, params=params if first else None, timeout=30)
        if resp.status_code != 200:
            print(f"Error {resp.status_code}: {resp.text[:300]}")
            resp.raise_for_status()
        payload = resp.json()
        records.extend(payload.get("value", []))
        url, first = payload.get("@odata.nextLink"), False
    return records


def to_naive_utc(series: pd.Series) -> pd.Series:
    """
    Parse mixed ISO8601 strings (date-only, with time, 7-digit fractional seconds, 'Z')
    into tz-naive UTC timestamps.

    format='ISO8601' makes pandas parse each value independently. Without it,
    pandas infers the format from the first value and turns every row that
    doesn't match into NaT.
    """
    return pd.to_datetime(series, errors="coerce", utc=True, format="ISO8601").dt.tz_localize(None)


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Accounts
# ─────────────────────────────────────────────────────────────────────────────

def build_accounts_df(records: list) -> pd.DataFrame:
    raw_df = pd.DataFrame(records).reset_index(drop=True)

    def _as_dict(x):
        return x if isinstance(x, dict) else {}

    cf_df = pd.json_normalize(raw_df["Cf"].apply(_as_dict).tolist())
    cf_df.index = raw_df.index

    if "UserAccount" in raw_df.columns:
        user_account_df = pd.json_normalize(raw_df["UserAccount"].apply(_as_dict).tolist())
    else:
        user_account_df = pd.DataFrame(index=raw_df.index)
    user_account_df.index = raw_df.index
    if "Name" not in user_account_df.columns:
        user_account_df["Name"] = np.nan
    user_account_df = user_account_df.rename(columns={"Name": "UserAccount_Name"})

    df_accounts = pd.concat(
        [
            raw_df[["Id", "ExternalId", "Name", "TotalContractAmount", "StartDate"]],
            user_account_df[["UserAccount_Name"]],
            cf_df,
        ],
        axis=1,
    )
    print("df_accounts shape:", df_accounts.shape)
    return df_accounts


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: "Starting MRR" notes
# ─────────────────────────────────────────────────────────────────────────────

def extract_mrr_value(html_content):
    """Pull the first $ amount out of an HTML note body."""
    if not isinstance(html_content, str):
        return None
    text = BeautifulSoup(html_content, "html.parser").get_text()
    match = re.search(r"\$\s*([0-9,.]+)", text)
    if match:
        try:
            return float(match.group(1).replace(",", ""))
        except ValueError:
            return None
    return None


def build_notes_df(note_records: list) -> pd.DataFrame:
    df_notes = pd.DataFrame(note_records)
    print(f"Fetched {len(df_notes)} notes")
    if df_notes.empty or "Content" not in df_notes.columns:
        return pd.DataFrame(columns=["AccountId", "Starting MRR"])
    df_notes["Starting MRR"] = df_notes["Content"].apply(extract_mrr_value)
    return df_notes[["AccountId", "Starting MRR"]]


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Merge and compute metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(df_accounts: pd.DataFrame, df_notes: pd.DataFrame) -> pd.DataFrame:
    df_accounts_clean = df_accounts.copy()
    df_accounts_clean["Id_str"] = df_accounts_clean["Id"].astype(str).str.strip()

    df_notes_clean = df_notes.copy()
    df_notes_clean["AccountId_str"] = df_notes_clean["AccountId"].astype(str).str.strip()

    # Several 'Starting MRR' notes per account: keep the first so the merge can't duplicate accounts
    df_notes_clean = df_notes_clean.drop_duplicates(subset="AccountId_str", keep="first")

    df_combined = pd.merge(
        df_accounts_clean,
        df_notes_clean[["AccountId_str", "Starting MRR"]],
        left_on="Id_str",
        right_on="AccountId_str",
        how="left",
    ).drop(columns=["Id_str", "AccountId_str"])

    df_combined = df_combined.drop(columns=["FunnelEnterDate"], errors="ignore")

    # A date field can be absent from every account's custom fields; treat as all-missing
    for col in DATE_COLUMNS:
        if col not in df_combined.columns:
            df_combined[col] = np.nan

    start_date = to_naive_utc(df_combined["StartDate"])
    picked_up_date = to_naive_utc(df_combined["CustomerPickedUpByOeDate"])
    qualified_date = to_naive_utc(df_combined["CustomerQualifiedByOeDate"])
    unqualified_date = to_naive_utc(df_combined["CustomerUnqualifiedByOeDate"])

    failed = df_combined["StartDate"].notna() & start_date.isna()
    print("StartDate values that failed to parse:", int(failed.sum()))
    if failed.any():
        print(df_combined.loc[failed, ["ExternalId", "StartDate"]].head(20).to_string())

    # Compare on calendar days only, so a StartDate with a time component vs a
    # date-only column gives 0 rather than -1
    start_day = start_date.dt.normalize()
    picked_day = picked_up_date.dt.normalize()
    qualified_day = qualified_date.dt.normalize()
    unqualified_day = unqualified_date.dt.normalize()

    df_combined["cf_TimeToPickupinDays"] = (picked_day - start_day).dt.days

    first_outcome = pd.concat([qualified_day, unqualified_day], axis=1).min(axis=1)  # NaT-safe
    df_combined["cf_TimeToQualifyinDays"] = (first_outcome - start_day).dt.days

    # Backfill pickup time with qualify time when pickup is missing
    mask = df_combined["cf_TimeToPickupinDays"].isna() & df_combined["cf_TimeToQualifyinDays"].notna()
    df_combined.loc[mask, "cf_TimeToPickupinDays"] = df_combined.loc[mask, "cf_TimeToQualifyinDays"]
    print("Rows backfilled:", int(mask.sum()))

    df_combined["StartDate"] = start_day.dt.strftime("%Y-%m-%d")

    total_contract = pd.to_numeric(df_combined["TotalContractAmount"], errors="coerce")
    starting_mrr = pd.to_numeric(df_combined["Starting MRR"], errors="coerce")
    df_combined["cf_MrrGrowth"] = total_contract - starting_mrr

    print("df_combined shape:", df_combined.shape)
    print("Accounts with missing StartDate:", int(df_combined["StartDate"].isna().sum()),
          "(should only be accounts with no StartDate in ChurnZero)")
    return df_combined


# ─────────────────────────────────────────────────────────────────────────────
# Step 4: Final dataset + push
# ─────────────────────────────────────────────────────────────────────────────

def build_final(df_combined: pd.DataFrame) -> pd.DataFrame:
    df_final = (
        df_combined[[
            "ExternalId",
            "Starting MRR",
            "cf_TimeToPickupinDays",
            "cf_TimeToQualifyinDays",
            "cf_MrrGrowth",
        ]]
        .rename(columns={"ExternalId": "AccountExternalId", "Starting MRR": "cf_StartingMrr"})
        .copy()
    )
    df_final = df_final.dropna(subset=["AccountExternalId"])
    df_final = df_final[df_final["AccountExternalId"].astype(str).str.strip() != ""]
    return df_final


def push_to_churnzero(csv_path: str):
    url = f"{CZ_BASE_URL}/batchAccountsCsv?appKey={CZ_IMPORT_APP_KEY}"
    with open(csv_path, "rb") as f:
        files = {"file": (csv_path, f, "text/csv")}
        resp = requests.post(url, files=files, timeout=120)
    print(f"Status Code: {resp.status_code}")
    print(f"Response: {resp.text}")
    resp.raise_for_status()  # let the job fail visibly
    print("File successfully posted!")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    records = cz_get_all("Account", params={"$filter": CZ_SEGMENT_FILTER})
    if not records:
        sys.exit("No accounts found")

    df_accounts = build_accounts_df(records)

    note_records = cz_get_all("Note", params={"$filter": "contains(Name, 'Starting MRR')"})
    df_notes = build_notes_df(note_records)

    df_combined = compute_metrics(df_accounts, df_notes)
    df_final = build_final(df_combined)

    df_final.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved {OUTPUT_CSV}", df_final.shape)

    push_to_churnzero(OUTPUT_CSV)


if __name__ == "__main__":
    main()
