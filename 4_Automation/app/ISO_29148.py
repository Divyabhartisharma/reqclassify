import os
from supabase import create_client
from google import genai

# keys are taken from env variables
supabase = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

# path to iso 29148 file
base = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
doc_path = os.path.join(base, "data", "iso29148_definitions.txt")

# info saved with every chunk in the table
doc_id = "iso29148_definitions"
doc_title = "ISO/IEC/IEEE 29148 Requirement Type Reference"
standard = "iso29148"
embed_model = "models/gemini-embedding-001"


# remove bad characters, otherwise postgres error for \x00
def clean_text(text):
    text = text.replace("\x00", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")   # windows new lines to normal
    return text.strip()


# split the file on "## " headings, every heading = one chunk
def make_chunks(text):
    chunks = []
    cur = []
    for line in text.splitlines():
        # new heading -> save old chunk and start new one
        if line.startswith("## ") and cur:
            chunks.append("\n".join(cur).strip())
            cur = []
        cur.append(line)
    # last chunk
    if cur:
        chunks.append("\n".join(cur).strip())

    # keep only real sections, very small ones are not useful for rag
    good = []
    for c in chunks:
        if c.startswith("## ") and len(c) > 200:
            good.append(c)
    return good


# delete only the old iso29148 rows, iso25010 rows in the table
supabase.table("ISO_Chunks").delete().eq("doc_id", doc_id).execute()

# reading the file
f = open(doc_path, "r", encoding="utf-8", errors="ignore")
text = clean_text(f.read())
f.close()

chunks = make_chunks(text)
print("chunks:", len(chunks))

# embedding for every chunk with gemini
rows = []
for i in range(len(chunks)):
    emb = client.models.embed_content(model=embed_model, contents=chunks[i]).embeddings[0].values
    if i == 0:
        print("embedding size:", len(emb))   # should be 3072

    # one row for the table
    rows.append({
        "doc_id": doc_id,
        "doc_title": doc_title,
        "standard_type": standard,
        "chunk_index": i,
        "content": chunks[i],
        "metadata": {"source": "iso29148", "purpose": "rag_grounding"},
        "embedding": emb
    })

# insert all rows at once
if len(rows) > 0:
    supabase.table("ISO_Chunks").insert(rows).execute()
    print("inserted", len(rows), "iso29148 chunks")
else:
    print("no chunks, check the file (headings should start with ## )")