from typing import List
from llama_index.core.schema import NodeWithScore
from llama_index.llms.groq import Groq

from backend.config import settings
from backend.models.schemas import SourceCitation, QueryResponse


class GroundedGenerator:

    def __init__(self):
        self.groq_api_key = settings.GROQ_API_KEY

    def _get_llm_instance(self, model_name: str) -> Groq:
        return Groq(
            model=model_name,
            api_key=self.groq_api_key,
            temperature=0.1,
        )

    def generate_answer(
        self, query: str, context_nodes: List[NodeWithScore], model_name: str, route_reason: str
    ) -> QueryResponse:
        if not context_nodes:
            return QueryResponse(
                answer="Insufficient information found in the knowledge base to answer the query.",
                citations=[],
                model_used=model_name,
                route_reasoning=route_reason,
            )

        formatted_context_blocks = []
        citations: List[SourceCitation] = []

        for idx, node_with_score in enumerate(context_nodes, start=1):
            node = node_with_score.node
            score = round(float(node_with_score.score or 0.0), 3)

            file_name = node.metadata.get("file_name", "Unknown Source")
            page_label = str(node.metadata.get("page_label", "N/A"))
            text_content = node.get_content().strip()

            formatted_context_blocks.append(
                f"--- SOURCE [{idx}] (File: {file_name}, Page: {page_label}) ---\n{text_content}"
            )

            citations.append(
                SourceCitation(
                    document_name=file_name,
                    page_label=page_label,
                    snippet=text_content[:200] + "..." if len(text_content) > 200 else text_content,
                    score=score,
                )
            )

        full_context = "\n\n".join(formatted_context_blocks)

        system_prompt = (
            "You are an enterprise knowledge base assistant.\n"
            "RESPONSE GUIDELINES:\n"
            "1. Answer strictly based on the provided Context.\n"
            "2. If the context lacks direct answers, output: 'Insufficient information in the knowledge base.' Do not extrapolate.\n"
            "3. Cite sources explicitly using inline notation format like [1], [2] when referencing facts.\n"
            "4. Maintain a professional and objective corporate tone."
        )

        user_prompt = f"CONTEXT:\n{full_context}\n\nUSER QUERY: {query}"

        llm = self._get_llm_instance(model_name)
        response = llm.complete(f"{system_prompt}\n\n{user_prompt}")

        return QueryResponse(
            answer=response.text,
            citations=citations,
            model_used=model_name,
            route_reasoning=route_reason,
        )