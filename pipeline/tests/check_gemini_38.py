import os
import dotenv
from google import genai

dotenv.load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

print("=" * 80)
print("QUERYING LIVE GEMINI API: client.models.list()")
print("=" * 80)

all_models = list(client.models.list())
print(f"Total models returned: {len(all_models)}")

flash_models = [m for m in all_models if "flash" in m.name.lower()]
print("\n--- Flash Models Found in Live Catalog ---")
for m in flash_models:
    print(f"  - Name: {m.name} | Display: {getattr(m, 'display_name', 'N/A')}")

target_38 = [m for m in all_models if "3.8" in m.name.lower()]
print("\n--- '3.8' Models Found ---")
if target_38:
    for m in target_38:
        print(f"  - Name: {m.name} | Display: {getattr(m, 'display_name', 'N/A')}")
else:
    print("  None found with '3.8' in name.")

print("\n--- Live generate_content test with 'gemini-3.8-flash' ---")
try:
    resp = client.models.generate_content(
        model="gemini-3.8-flash",
        contents="Say hello in one word."
    )
    print(f"SUCCESS: {resp.text.strip()}")
except Exception as e:
    print(f"EXCEPTION TYPE: {type(e).__name__}")
    print(f"EXCEPTION DETAILS: {e}")

print("=" * 80)
