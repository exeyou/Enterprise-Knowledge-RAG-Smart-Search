import json
from typing import Tuple
from llama_index.llms.groq import Groq
from backend.config import settings


class ModelRouter:

    def __init__(self):
        self.classifier_llm = Groq(
            model="llama-3.1-8b-instant",
            api_key=settings.GROQ_API_KEY,
            temperature=0.0
        )

    def route_query(self, query: str, force_flagship: bool = False) -> Tuple[str, str]:
        if force_flagship:
            return (
                settings.FLAGSHIP_LLM_MODEL,
                "Forced flagship model execution selected by user profile.",
            )

        prompt = f"""
Analyze the incoming query directed at the enterprise knowledge base and determine processing complexity.

Classification Rules:
- FAST: Straightforward, factual query requiring simple metadata, term definition, short standard, or direct retrieval context match.
- FLAGSHIP: Complex inquiry requiring synthesis across multiple documents, multi-step logical deduction, causal analysis, or code generation.

Query: "{query}"

Respond STRICTLY in JSON format without markdown code blocks:
{{"complexity": "FAST" or "FLAGSHIP", "reason": "concise explanation within one sentence"}}
"""

        try:
            response = self.classifier_llm.complete(prompt).text.strip()
            if response.startswith("```json"):
                response = response[7:-3].strip()
            elif response.startswith("```"):
                response = response[3:-3].strip()

            result = json.loads(response)
            complexity = result.get("complexity", "FAST")
            reason = result.get("reason", "Automated routing executed successfully.")

            if complexity == "FLAGSHIP":
                return settings.FLAGSHIP_LLM_MODEL, f"[Flagship] {reason}"
            else:
                return settings.FAST_LLM_MODEL, f"[Fast] {reason}"

        except Exception as e:
            return (
                settings.FAST_LLM_MODEL,
                f"[Fallback] Executed fast model due to classification exception: {str(e)}",
            )