import spacy
from typing import List, Dict

class EntityExtractor:
    """
    Extrai entidades biomédicas usando SciSpaCy.
    Retorna um dicionário estruturado para futura construção do Knowledge Graph.
    """

    def __init__(self):
        # Modelo básico científico
        self.nlp_base = spacy.load("en_core_sci_sm")
        # Modelo biomédico avançado (genes, proteínas, compostos, células, pathways, etc.)
        self.nlp_bio = spacy.load("en_ner_bionlp13cg_md")

    def extract(self, text: str) -> Dict:
        if not text or len(text.strip()) == 0:
            return {"entities": [], "relations": []}

        # Primeiro passa pelo modelo científico básico
        doc = self.nlp_base(text)

        # Depois pelo modelo biomédico especializado
        bio_doc = self.nlp_bio(text)

        # Collect entities
        entities = []
        for ent in bio_doc.ents:
            entities.append({
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char
            })

        # Placeholder para relações (vamos implementar depois)
        relations = []

        return {
            "entities": entities,
            "relations": relations,
        }
