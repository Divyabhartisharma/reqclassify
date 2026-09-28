import os
from supabase import create_client
from google import genai

# keys are taken from env variables
supabase = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

# path to iso file, data folder is one folder up from this script
base = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
doc_path = os.path.join(base, "data", "iso_definitions.txt")

# info that is saved with every chunk in the table
doc_id = "iso25010_sections_v1"
doc_title = "ISO/IEC 25010 NFR Reference Sections"
standard = "iso25010"
embed_model = "models/gemini-embedding-001"   # gives 3072 numbers per text


# remove bad characters, postgres gives error for \x00
def clean_text(text):
    text = text.replace("\x00", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")   # windows new lines to normal
    return text.strip()


# split the file on "## " headings, every heading = one chunk
def make_chunks(text):
    chunks = []
    cur = []
    for line in text.splitlines():
        # new heading -> save the old chunk and start new one
        if line.startswith("## ") and cur:
            chunks.append("\n".join(cur).strip())
            cur = []
        cur.append(line)
    # last chunk
    if cur:
        chunks.append("\n".join(cur).strip())

    # only keep real sections
    good = []
    for c in chunks:
        if c.startswith("## ") and len(c) > 200:
            good.append(c)
    return good


# delete old rows of this doc first, for more duplicate values
supabase.table("ISO_Chunks").delete().eq("doc_id", doc_id).execute()

# read the iso file
f = open(doc_path, "r", encoding="utf-8", errors="ignore")
text = clean_text(f.read())
f.close()

chunks = make_chunks(text)
print("chunks:", len(chunks))

# make embedding for every chunk with gemini
rows = []
for i in range(len(chunks)):
    res = client.models.embed_content(model=embed_model, contents=chunks[i])
    emb = res.embeddings[0].values
    if i == 0:
        print("embedding size:", len(emb))   # should be 3072, table column also 3072

    # one row for supabase table
    rows.append({
        "doc_id": doc_id,
        "doc_title": doc_title,
        "standard_type": standard,
        "chunk_index": i,
        "content": chunks[i],
        "metadata": {"source": "iso25010", "purpose": "rag_grounding"},
        "embedding": emb
    })

# inserting rows
if len(rows) > 0:
    supabase.table("ISO_Chunks").insert(rows).execute()
    print("inserted", len(rows), "chunks")
else:
    print("no chunks, check the file format (headings should start with ## )")