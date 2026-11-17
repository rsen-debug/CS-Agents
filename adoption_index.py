"""
Daily Feature Adoption Index
-----------------------------
Pulls a ChurnZero account segment, computes each account's feature adoption
index (adopted features / eligible features, as a percentage), and pushes the
results back into ChurnZero as account custom fields.

Changes vs. the original notebook, needed to run unattended:

1. Credentials come from environment variables (GitHub Actions secrets)
   instead of being hardcoded.
2. A failed upload now fails the job. The notebook caught the exception and
   just printed it, which in a scheduled run would show a green check even
   when nothing was uploaded.
3. display() (notebook-only) replaced with print().
4. Feature columns are cast to object dtype before "Adopted" is written into
   them. Writing a string into an all-NaN float column warns on pandas 2.x
   and raises on pandas 3.x.
"""

import os
import sys

import numpy as np
import pandas as pd
import requests


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
CZ_SEGMENT_FILTER = os.environ.get("CZ_SEGMENT_FILTER") or "Use/SegmentId eq 3515 and Use/SegmentColumnSetId eq 197"

CZ_HEADERS = {
    "Authorization": CZ_AUTH_HEADER,
    "Content-Type": "application/json",
}

OUTPUT_CSV = "adoption_index.csv"

# feature key -> (usage column, feature adoption column)
FEATURE_MAPPING = {
    "cloudflare": ("DomainCount", "FeatureAdoptioncloudflare"),
    "malware": ("Sscount", "FeatureAdoptionmalware"),
    "site_manager": ("Sitemanagerappcount", "FeatureAdoptionsitemanager"),
    "managed_ai": ("Managedagentscount", "Featureadoptionmanagedai"),
    "mcp": ("Mcpconnectionscount", "FeatureadoptioncloudwaysMcp"),
    "dns_made_easy": ("Dmecount", "FeatureadoptionDnsmadeeasy"),
    "nodejs": ("Nodejsapps", "Featureadoptionnodejs"),
    "copilot": ("Copilotplancurrent", "FeatureadoptionCloudwaysCopilot"),
}

RENAME_MAPPING = {
    "FeatureAdoptioncloudflare": "cf_FeatureAdoptioncloudflare",
    "FeatureAdoptionmalware": "cf_FeatureAdoptionmalware",
    "FeatureAdoptionsitemanager": "cf_FeatureAdoptionsitemanager",
    "FeatureadoptioncloudwaysMcp": "cf_FeatureadoptioncloudwaysMcp",
    "FeatureadoptionDnsmadeeasy": "cf_FeatureadoptionDnsmadeeasy",
    "FeatureadoptionCloudwaysCopilot": "cf_FeatureadoptionCloudwaysCopilot",
    "Featureadoptionnodejs": "cf_Featureadoptionnodejs",
    "Featureadoptionmanagedai": "cf_Featureadoptionmanagedai",
    "Featureadoptionindex": "cf_Featureadoptionindex",
    "ExternalId": "AccountExternalId",
}


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Fetch the ChurnZero segment
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
    print(f"Fetched {len(df)} accounts")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Flatten custom fields
# ─────────────────────────────────────────────────────────────────────────────

def flatten_custom_fields(df: pd.DataFrame) -> pd.DataFrame:
    cf_series = df["Cf"].apply(lambda x: x if isinstance(x, dict) else {})
    cf_df = pd.json_normalize(cf_series.tolist())
    return pd.concat([df[["ExternalId"]].reset_index(drop=True), cf_df], axis=1)


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Compute adoption + index
# ─────────────────────────────────────────────────────────────────────────────

def check_adopted(row, feat, usage_col) -> bool:
    val = row.get(usage_col)
    if pd.isna(val):
        return False

    if feat == "copilot":
        # Adopted unless blank / None / Free Starter
        val_str = str(val).strip()
        return val_str not in ["", "None", "nan", "NaN", "Free Starter"]

    try:
        return float(val) > 0
    except (ValueError, TypeError):
        return False


def get_clean_feature_val(val) -> str:
    if isinstance(val, list):
        if len(val) == 0:
            return ""
        val = val[0]
    if pd.isna(val):
        return ""
    return str(val).strip()


def compute_adoption(df: pd.DataFrame) -> pd.DataFrame:
    for _, (_, feat_col) in FEATURE_MAPPING.items():
        if feat_col not in df.columns:
            df[feat_col] = np.nan
        df[feat_col] = df[feat_col].astype(object)

    indexes = []
    for idx, row in df.iterrows():
        eligible_count = 0
        active_count = 0

        for feat, (usage_col, feat_col) in FEATURE_MAPPING.items():
            current_feat_val = get_clean_feature_val(row.get(feat_col))

            # Ineligible only if the column says "Not Adopted" or "Not Applicable"
            is_eligible = not ("Not Adopted" in current_feat_val or "Not Applicable" in current_feat_val)

            if is_eligible:
                eligible_count += 1
                if check_adopted(row, feat, usage_col):
                    active_count += 1
                    df.at[idx, feat_col] = "Adopted"
                elif current_feat_val != "Adopted":
                    df.at[idx, feat_col] = np.nan
            else:
                df.at[idx, feat_col] = current_feat_val

        indexes.append(active_count / eligible_count if eligible_count > 0 else np.nan)

    df["Featureadoptionindex"] = np.array(indexes, dtype=float) * 100
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Step 4: Push to ChurnZero
# ─────────────────────────────────────────────────────────────────────────────

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
    df = fetch_churnzero_accounts()
    if df.empty or "Cf" not in df.columns or "ExternalId" not in df.columns:
        print("Nothing to process. Exiting.")
        return

    df = flatten_custom_fields(df)
    df = compute_adoption(df)

    cols_to_show = ["ExternalId", "Featureadoptionindex"] + [feat_col for _, feat_col in FEATURE_MAPPING.values()]
    print(df[cols_to_show].head(10).to_string())

    df = df.rename(columns=RENAME_MAPPING)
    output_cols = [RENAME_MAPPING.get(c, c) for c in cols_to_show]

    print(f"Rows before cleaning: {len(df)}")
    df = df[df["AccountExternalId"].notna() & (df["AccountExternalId"].astype(str).str.strip() != "")]
    print(f"Rows after cleaning: {len(df)}")

    df[output_cols].to_csv(OUTPUT_CSV, index=False)
    print(f"Saved {len(df)} rows to {OUTPUT_CSV}")

    push_to_churnzero(OUTPUT_CSV)


if __name__ == "__main__":
    main()
