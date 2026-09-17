import json
from pathlib import Path
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

token_path = Path("youtube_token.json")
if not token_path.exists():
    print("youtube_token.json does not exist")
    exit(0)

try:
    with open(token_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print("Token keys:", list(data.keys()))
    creds = Credentials.from_authorized_user_file(str(token_path))
    if creds.expired and creds.refresh_token:
        print("Refreshing expired token...")
        creds.refresh(Request())
        with open(token_path, "w", encoding="utf-8") as f:
            f.write(creds.to_json())
        print("Token refreshed successfully!")
    print("Valid:", creds.valid)
    youtube = build("youtube", "v3", credentials=creds)
    ch_res = youtube.channels().list(part="snippet,contentDetails", mine=True).execute()
    for ch in ch_res.get("items", []):
        print("Channel title:", ch["snippet"]["title"])
        uploads_id = ch["contentDetails"]["relatedPlaylists"]["uploads"]
        p_res = youtube.playlistItems().list(part="snippet", playlistId=uploads_id, maxResults=20).execute()
        items = p_res.get("items", [])
        print(f"Total videos on channel: {len(items)}")
        for item in items:
            vid = item["snippet"]["resourceId"]["videoId"]
            title = item["snippet"]["title"]
            print(f"  - https://youtube.com/shorts/{vid} ({title})")
except Exception as e:
    print("YouTube API Exception:", type(e), e)
