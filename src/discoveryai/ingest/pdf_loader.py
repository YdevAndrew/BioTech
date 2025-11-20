import fitz  # PyMuPDF
from typing import List, Dict
from discoveryai.utils.splitter import split_text

class PDFLoader:
    def load_chunks(self, path: str, chunk_size: int = 1000, overlap: int = 200) -> List[Dict]:
        doc = fitz.open(path)
        out = []
        for page_num, page in enumerate(doc, start=1):
            page_text = page.get_text().strip()
            if not page_text:
                continue

            chunks = split_text(page_text, chunk_size=chunk_size, overlap=overlap)

            for i, ch in enumerate(chunks):
                out.append({
                    "text": ch,
                    "metadata": {
                        "source": path,
                        "page": page_num,
                        "chunk_id": f"p{page_num}_c{i}",
                    }
                })

        return out
