"""YouTube OAuth Interactive Re-Authentication Script.

Prints the exact Google authorization URL and opens it in the default browser.
Once approved, captures the credentials and saves them to youtube_token.json.
"""

import sys
import webbrowser
from pathlib import Path
from google_auth_oauthlib.flow import InstalledAppFlow

ROOT_DIR = Path(__file__).resolve().parent

YOUTUBE_SCOPES = [
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube",
    "https://www.googleapis.com/auth/youtube.readonly",
    "https://www.googleapis.com/auth/yt-analytics.readonly",
]

def main():
    client_secrets = ROOT_DIR / "client_secrets.json"
    token_path = ROOT_DIR / "youtube_token.json"

    if not client_secrets.exists():
        print(f"Error: {client_secrets} not found!", flush=True)
        return

    print("=" * 70, flush=True)
    print("Initiating YouTube OAuth flow...", flush=True)
    print("=" * 70, flush=True)

    flow = InstalledAppFlow.from_client_secrets_file(str(client_secrets), YOUTUBE_SCOPES)
    auth_url, _ = flow.authorization_url(prompt='consent', access_type='offline')

    print("\nPlease open this URL in your browser to approve YouTube access:\n", flush=True)
    print(auth_url, flush=True)
    print("\n" + "=" * 70, flush=True)

    try:
        webbrowser.open(auth_url)
    except Exception as e:
        print(f"Could not open browser automatically: {e}", flush=True)

    print("Waiting for authorization callback on localhost...", flush=True)
    creds = flow.run_local_server(port=0, open_browser=False)

    with open(token_path, "w", encoding="utf-8") as f:
        f.write(creds.to_json())

    print(f"\n[SUCCESS] Fresh YouTube OAuth token saved to {token_path.name}!", flush=True)

if __name__ == "__main__":
    main()
