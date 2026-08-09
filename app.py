import os
import streamlit as st
import requests

st.set_page_config(
    page_title="Enterprise Knowledge RAG",
    page_icon="🔍",
    layout="wide"
)

API_BASE_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000/api/v1")

st.title("Enterprise Knowledge RAG & Smart Search")
st.caption("Context-grounded enterprise documentation lookup engine with LLM routing optimization.")

with st.sidebar:
    st.header("Document Ingestion")
    uploaded_files = st.file_uploader(
        "Upload source files (PDF, MD, TXT):",
        type=["pdf", "md", "txt"],
        accept_multiple_files=True
    )

    if st.button("Index Documents", type="primary"):
        if uploaded_files:
            files_to_send = [
                ("files", (file.name, file.getvalue(), file.type)) for file in uploaded_files
            ]
            with st.spinner("Executing document parsing, chunking, and vector embedding..."):
                try:
                    res = requests.post(f"{API_BASE_URL}/ingest", files=files_to_send)
                    if res.status_code == 200:
                        data = res.json()
                        st.success(f"Successfully processed files: {len(data['processed_files'])}")
                        st.info(f"Total chunks registered: {data['total_chunks_added']}")
                    else:
                        st.error(f"Server error response: {res.text}")
                except Exception as e:
                    st.error(f"Backend connection failure: {e}")
        else:
            st.warning("Please attach at least one valid target file.")

    st.divider()
    st.header("Query Parameters")
    force_flagship = st.checkbox("Force Flagship LLM Tier", value=False)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "citations" in message and message["citations"]:
            with st.expander("View Grounded Citations & Sources"):
                for idx, cit in enumerate(message["citations"], 1):
                    st.write(
                        f"**[{idx}] {cit['document_name']}** (Page {cit['page_label']}) — *Relevance Score: {cit['score']}*")
                    st.caption(f'"{cit["snippet"]}"')
        if "model_info" in message:
            st.caption(f"Model: `{message['model_info']}` | {message.get('reason', '')}")

if prompt := st.chat_input("Query enterprise knowledge documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Executing hybrid vector search and semantic routing..."):
            try:
                payload = {
                    "query": prompt,
                    "top_k": 5,
                    "force_flagship": force_flagship
                }
                response = requests.post(f"{API_BASE_URL}/query", json=payload)

                if response.status_code == 200:
                    data = response.json()
                    answer = data["answer"]
                    citations = data["citations"]
                    model_used = data["model_used"]
                    reasoning = data["route_reasoning"]

                    st.markdown(answer)

                    if citations:
                        with st.expander("View Grounded Citations & Sources"):
                            for idx, cit in enumerate(citations, 1):
                                st.write(
                                    f"**[{idx}] {cit['document_name']}** (Page {cit['page_label']}) — *Relevance Score: {cit['score']}*")
                                st.caption(f'"{cit["snippet"]}"')

                    st.caption(f"Model: `{model_used}` | {reasoning}")

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "citations": citations,
                        "model_info": model_used,
                        "reason": reasoning
                    })
                else:
                    st.error(f"Failed to fetch execution response: {response.text}")
            except Exception as e:
                st.error(f"Service connection exception: {e}")