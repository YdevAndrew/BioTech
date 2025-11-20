from typing import List

def split_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Divide `text` em chunks com tamanho chunk_size (tokens ~ chars here) e overlap.
    Simplesmente baseado em caracteres; substitua por tokenização caso precise precisão.
    """
    if not text:
        return []

    text = text.replace("\r\n", "\n").strip()
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk.strip())
        start = end - overlap
        if start < 0:
            start = 0
    return [c for c in chunks if c]