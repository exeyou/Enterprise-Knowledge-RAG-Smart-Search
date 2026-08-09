import os
from typing import List
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document, TextNode
from llama_index.readers.file import PyMuPDFReader


class DocumentIngestor:

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.splitter = SentenceSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        self.pdf_reader = PyMuPDFReader()

    def process_file(self, file_path: str) -> List[TextNode]:
        file_name = os.path.basename(file_path)
        ext = os.path.splitext(file_name)[1].lower()

        raw_documents: List[Document] = []

        if ext == ".pdf":
            raw_documents = self.pdf_reader.load_data(file_path)
            for doc in raw_documents:
                doc.metadata["file_name"] = file_name
                doc.metadata["file_type"] = "pdf"
        elif ext in [".md", ".txt"]:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            doc = Document(
                text=content,
                metadata={
                    "file_name": file_name,
                    "file_type": ext.replace(".", ""),
                    "page_label": "1",
                },
            )
            raw_documents.append(doc)
        else:
            raise ValueError(f"Unsupported file extension: {ext}")

        nodes = self.splitter.get_nodes_from_documents(raw_documents)

        for node in nodes:
            node.metadata["file_name"] = file_name
            if "page_label" not in node.metadata:
                node.metadata["page_label"] = "1"

        return nodes