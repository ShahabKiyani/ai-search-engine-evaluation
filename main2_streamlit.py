import pickle
import re
import urllib.parse

import streamlit as st
from beir.datasets.data_loader import GenericDataLoader
from txtai import Embeddings

DATASET_PATH = "datasets/nq"


def normalize_query(text):
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def compute_ir_metrics(retrieved_doc_ids, relevant_doc_ids, k):
    retrieved_at_k = retrieved_doc_ids[:k]
    relevant_set = set(relevant_doc_ids)

    true_positives = sum(1 for doc_id in retrieved_at_k if doc_id in relevant_set)

    precision = true_positives / k if k > 0 else 0.0
    recall = true_positives / len(relevant_set) if relevant_set else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return precision, recall, f1, true_positives


def first_relevant_rank(retrieved_doc_ids, relevant_doc_ids):
    relevant_set = set(relevant_doc_ids)

    for rank, doc_id in enumerate(retrieved_doc_ids, start=1):
        if doc_id in relevant_set:
            return rank

    return None


@st.cache_resource
def load_data_and_embeddings():
    corpus, queries, qrels = GenericDataLoader(data_folder=DATASET_PATH).load(split="test")

    with open("nq_doc_ids.pkl", "rb") as f:
        doc_ids = pickle.load(f)

    embeddings = Embeddings({
        "path": "sentence-transformers/all-MiniLM-L6-v2"
    })
    embeddings.load("nq_embeddings")

    normalized_query_to_id = {
        normalize_query(query_text): query_id
        for query_id, query_text in queries.items()
    }

    return corpus, queries, qrels, doc_ids, embeddings, normalized_query_to_id


# Load everything
corpus, queries, qrels, doc_ids, embeddings, normalized_query_to_id = load_data_and_embeddings()

st.title("Search Engine - BEIR NQ")

query = st.text_input("Enter your search query:", "")

if st.button("Search"):
    if query.strip():

        # Retrieve top 20 results
        results = embeddings.search(query, 20)

        raw_retrieved_doc_ids = []
        retrieved_docs = []

        for idx, score in results:
            doc_id = doc_ids[idx]
            doc = corpus[doc_id]

            title = doc.get("title", "") or "(No title)"
            text = doc.get("text", "") or ""

            wiki_url = "https://en.wikipedia.org/wiki/" + urllib.parse.quote(title.replace(" ", "_"))

            raw_retrieved_doc_ids.append(doc_id)

            retrieved_docs.append({
                "doc_id": doc_id,
                "title": title,
                "text": text[:400],
                "url": wiki_url,
                "score": score
            })

        # Match query to BEIR query
        normalized_input = normalize_query(query)
        matched_query_id = normalized_query_to_id.get(normalized_input)

        metrics_data = None
        relevant_doc_ids = []

        if matched_query_id is not None:
            relevant_doc_ids = [
                doc_id for doc_id, rel in qrels[matched_query_id].items()
                if rel > 0
            ]

            p5, r5, f1_5, _ = compute_ir_metrics(raw_retrieved_doc_ids, relevant_doc_ids, 5)
            p10, r10, f1_10, _ = compute_ir_metrics(raw_retrieved_doc_ids, relevant_doc_ids, 10)
            p20, r20, f1_20, _ = compute_ir_metrics(raw_retrieved_doc_ids, relevant_doc_ids, 20)
            first_hit_rank = first_relevant_rank(raw_retrieved_doc_ids, relevant_doc_ids)

            metrics_data = {
                "query_id": matched_query_id,
                "num_relevant": len(relevant_doc_ids),
                "p5": p5,
                "r5": r5,
                "f1_5": f1_5,
                "p10": p10,
                "r10": r10,
                "f1_10": f1_10,
                "p20": p20,
                "r20": r20,
                "f1_20": f1_20,
                "first_hit_rank": first_hit_rank
            }

        # ---- IR Evaluation ----
        st.markdown("---")
        st.subheader("IR Evaluation")

        if metrics_data:
            st.caption(
                f"Matched BEIR query ID: {metrics_data['query_id']} | Relevant docs: {metrics_data['num_relevant']}"
            )

            col1, col2, col3 = st.columns(3)
            col1.metric("Precision@5", f"{metrics_data['p5']:.3f}")
            col2.metric("Recall@5", f"{metrics_data['r5']:.3f}")
            col3.metric("F1@5", f"{metrics_data['f1_5']:.3f}")

            col4, col5, col6 = st.columns(3)
            col4.metric("Precision@10", f"{metrics_data['p10']:.3f}")
            col5.metric("Recall@10", f"{metrics_data['r10']:.3f}")
            col6.metric("F1@10", f"{metrics_data['f1_10']:.3f}")

            col7, col8, col9 = st.columns(3)
            col7.metric("Precision@20", f"{metrics_data['p20']:.3f}")
            col8.metric("Recall@20", f"{metrics_data['r20']:.3f}")
            col9.metric(
                "First Relevant Rank",
                str(metrics_data["first_hit_rank"]) if metrics_data["first_hit_rank"] else "None"
            )

        else:
            st.info("No matching BEIR query → metrics not available.")

        # ---- Search Results ----
        st.markdown("---")
        st.subheader("Search Results (Raw Ranking)")

        for rank, doc in enumerate(retrieved_docs, start=1):
            st.markdown(f"### {rank}. [{doc['title']}]({doc['url']})")
            st.caption(f"{doc['url']} | Score: {doc['score']:.4f} | Doc ID: {doc['doc_id']}")
            st.write(doc["text"] + ("..." if len(doc["text"]) == 400 else ""))
            st.divider()

    else:
        st.write("Please enter a search query.")
