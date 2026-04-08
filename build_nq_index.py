from beir.datasets.data_loader import GenericDataLoader
from txtai import Embeddings
import pickle
import os

DATASET_PATH = "datasets/nq"

print("Starting script...")
print("Current working directory:", os.getcwd())
print("Looking for dataset at:", os.path.abspath(DATASET_PATH))
print("Exists?", os.path.exists(DATASET_PATH))

# Load BEIR NQ
print("Loading corpus...")
corpus, queries, qrels = GenericDataLoader(data_folder=DATASET_PATH).load(split="test")
print("Loaded corpus:", len(corpus))
print("Loaded queries:", len(queries))
print("Loaded qrels:", len(qrels))
# Keep a stable order
doc_ids = list(corpus.keys())
print("Using docs:", len(doc_ids))

# Build searchable documents
documents = []
for i, doc_id in enumerate(doc_ids):
    title = corpus[doc_id].get("title", "") or ""
    text = corpus[doc_id].get("text", "") or ""
    doc = f"{title}. {text}".strip()
    documents.append(doc)

    if i % 10000 == 0:
        print(f"Prepared {i} documents...")

print("Building embeddings object...")
embeddings = Embeddings({
    "path": "sentence-transformers/all-MiniLM-L6-v2"
})

print("Indexing documents...")
embeddings.index(documents)

print("Saving embeddings...")
embeddings.save("nq_embeddings")

print("Saving doc ids...")
with open("nq_doc_ids.pkl", "wb") as f:
    pickle.dump(doc_ids, f)


print("Loaded corpus:", len(corpus))

print("Using docs:", len(doc_ids))


print("Done.")
print(f"Indexed {len(documents)} documents")