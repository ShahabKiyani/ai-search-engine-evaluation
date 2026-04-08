import pickle
import re
import urllib.parse

import ollama
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


def build_context(docs):
    context = ""

    for i, doc in enumerate(docs, start=1):
        context += f"""
Document {i}
Title: {doc['title']}
Text: {doc['text']}
"""
    return context


def generate_ai_answer(query, docs):
    context = build_context(docs)

    prompt = f"""
You are an AI search engine.

Answer the user's question using ONLY the information in the provided documents.

Rules:
- Do NOT mention documents, sources, or how the information was retrieved
- Do NOT say "according to the documents" or reference document numbers
- Do NOT mention if something is missing or not explicitly stated
- Provide a clear, natural answer as if you already know the information
- If needed, infer a reasonable answer from the context
- Keep the response to one concise paragraph

Question:
{query}

Documents:
{context}
"""

    response = ollama.chat(
        model="llama3.2",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response["message"]["content"]


corpus, queries, qrels, doc_ids, embeddings, normalized_query_to_id = load_data_and_embeddings()

st.title("Search Engine - BEIR NQ")

query = st.text_input("Enter your search query:", "")

if st.button("Search"):
    if query.strip():
        # Use top 20 raw results for evaluation and display
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

        normalized_input = normalize_query(query)
        matched_query_id = normalized_query_to_id.get(normalized_input)

        metrics_data = None
        relevant_doc_ids = []

        if matched_query_id is not None:
            relevant_doc_ids = [
                doc_id for doc_id, rel in qrels[matched_query_id].items()
                if rel > 0
            ]

            p5, r5, f1_5, tp5 = compute_ir_metrics(raw_retrieved_doc_ids, relevant_doc_ids, 5)
            p10, r10, f1_10, tp10 = compute_ir_metrics(raw_retrieved_doc_ids, relevant_doc_ids, 10)
            p20, r20, f1_20, tp20 = compute_ir_metrics(raw_retrieved_doc_ids, relevant_doc_ids, 20)
            first_hit_rank = first_relevant_rank(raw_retrieved_doc_ids, relevant_doc_ids)

            metrics_data = {
                "query_id": matched_query_id,
                "num_relevant": len(relevant_doc_ids),
                "p5": p5,
                "r5": r5,
                "f1_5": f1_5,
                "tp5": tp5,
                "p10": p10,
                "r10": r10,
                "f1_10": f1_10,
                "tp10": tp10,
                "p20": p20,
                "r20": r20,
                "f1_20": f1_20,
                "tp20": tp20,
                "first_hit_rank": first_hit_rank
            }

        if retrieved_docs:
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
                    str(metrics_data["first_hit_rank"]) if metrics_data["first_hit_rank"] is not None else "None"
                )

                with st.expander("Debug IR Data"):
                    st.write("Matched query ID:", matched_query_id)
                    st.write("Matched BEIR query text:", queries[matched_query_id])

                    st.write("Top retrieved doc IDs (top 20):", raw_retrieved_doc_ids)
                    st.write("Relevant doc IDs:", relevant_doc_ids)

                    overlap_top_5 = [doc_id for doc_id in raw_retrieved_doc_ids[:5] if doc_id in set(relevant_doc_ids)]
                    overlap_top_10 = [doc_id for doc_id in raw_retrieved_doc_ids[:10] if doc_id in set(relevant_doc_ids)]
                    overlap_top_20 = [doc_id for doc_id in raw_retrieved_doc_ids[:20] if doc_id in set(relevant_doc_ids)]

                    st.write("Overlap in top 5:", overlap_top_5)
                    st.write("Overlap in top 10:", overlap_top_10)
                    st.write("Overlap in top 20:", overlap_top_20)

                    retrieved_title_debug = []
                    for doc_id in raw_retrieved_doc_ids:
                        if doc_id in corpus:
                            retrieved_title_debug.append({
                                "doc_id": doc_id,
                                "title": corpus[doc_id].get("title", "")
                            })

                    relevant_title_debug = []
                    for doc_id in relevant_doc_ids:
                        if doc_id in corpus:
                            relevant_title_debug.append({
                                "doc_id": doc_id,
                                "title": corpus[doc_id].get("title", "")
                            })

                    st.write("Retrieved doc titles:", retrieved_title_debug)
                    st.write("Relevant doc titles:", relevant_title_debug)
            else:
                st.info("This query does not exactly match a BEIR NQ query, so qrels-based metrics are not shown.")

            st.subheader("AI Output")
            with st.spinner("Generating AI answer..."):
                try:
                    ai_answer = generate_ai_answer(query, retrieved_docs[:5])
                    st.write(ai_answer)
                except Exception as e:
                    st.warning(f"The AI response is temporarily unavailable: {e}")

            st.markdown("---")
            st.subheader("Search Results (Raw Ranking)")

            for rank, doc in enumerate(retrieved_docs, start=1):
                st.markdown(f"### {rank}. [{doc['title']}]({doc['url']})")
                st.caption(f"{doc['url']} | Score: {doc['score']:.4f} | Doc ID: {doc['doc_id']}")
                st.write(doc["text"] + ("..." if len(doc["text"]) == 400 else ""))
                st.divider()
        else:
            st.write("No results found.")
    else:
        st.write("Please enter a search query.")