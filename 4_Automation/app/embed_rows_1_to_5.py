from google import genai
from supabase import create_client
import os

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_KEY"]
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]

ROW_IDS = [1, 2, 3, 4, 5]

genai_client = genai.Client(api_key=GEMINI_API_KEY)
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

rows = supabase.table("nfr_examples") \
    .select('id, "Requirement"') \
    .in_("id", ROW_IDS) \
    .execute()

print(f"Fetched {len(rows.data)} rows")

for row in rows.data:
    text = row["Requirement"]

    response = genai_client.models.embed_content(
        model="models/embedding-001",
        contents=text
    )

    embedding = response.embeddings[0].values

    supabase.table("nfr_examples") \
        .update({"Embeddings": embedding}) \
        .eq("id", row["id"]) \
        .execute()

    print(f"Embedded row id={row['id']}")
