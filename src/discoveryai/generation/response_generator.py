import os
import requests
from typing import List

class ResponseGenerator:
    """
    Usa BioMistral para gerar respostas médicas usando o contexto selecionado.
    """

    def __init__(self):
        self.API_KEY = os.getenv("HF_API_KEY")
        self.MODEL = "BioMistral/BioMistral-7B"

        if not self.API_KEY:
            raise ValueError("Defina a variável de ambiente HF_API_KEY")

    def _build_prompt(self, query: str, context_chunks: List[str]):
        context_text = "\n\n".join(context_chunks)

        prompt = f"""
You are an expert biomedical assistant.  
Answer the question using ONLY the context below.  
If the context does not contain the answer, say you don't know.

### Context:
{context_text}

### Question:
{query}

### Answer:
"""
        return prompt

    def generate(self, query: str, context_chunks: List[str]) -> str:
        prompt = self._build_prompt(query, context_chunks)

        payload = {
            "inputs": prompt,
            "parameters": {
                "temperature": 0.0,
                "max_new_tokens": 500,
            }
        }

        response = requests.post(
            f"https://api-inference.huggingface.co/models/{self.MODEL}",
            headers={"Authorization": f"Bearer {self.API_KEY}"},
            json=payload
        )

        if response.status_code != 200:
            raise RuntimeError(f"HF API Error: {response.text}")

        data = response.json()
        return data[0]["generated_text"].replace(prompt, "").strip()
