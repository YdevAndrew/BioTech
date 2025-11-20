import os
from fastapi import FastAPI, UploadFile, File, Form, HTTPException

from discoveryai.nlp import reranker
from discoveryai.vectorstore import faiss_store

# cria diretório tmp/ se não existir
os.makedirs("tmp", exist_ok=True)

from discoveryai.ingest.pdf_loader import PDFLoader
from discoveryai.embeddings.embedder import Embedder
from discoveryai.vectorstore.faiss_store import FaissStore
from discoveryai.ner.extractor import EntityExtractor
from discoveryai.search.bm25_search import BM25Search
from discoveryai.services.vector_service import VectorService
from discoveryai.generation.response_generator import ResponseGenerator



# -----------------------------
# Inicialização de componentes
# -----------------------------

app = FastAPI(title="Discovery AI API")
UPLOAD_DIR = os.path.join(os.getcwd(), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

generator = ResponseGenerator()
loader = PDFLoader()
embedder = Embedder()
vector_service = VectorService()   # <-- CORRETO (classe instanciada)
VECTOR_DIM = 384
vectorstore = FaissStore(dim=VECTOR_DIM)
ner = EntityExtractor()
bm25 = BM25Search()

# REMOVE ESTA LINHA (INCORRETA E QUEBRAVA O SERVIDOR)
# result = vector_service.semantic_search(q, k=k)
# Essa linha usava q e k que não existem e era executada na importação, causando o erro.


# -----------------------------
# Rotas
# -----------------------------

@app.get("/")
def root():
    return {"message": "API DiscoveryAI funcionando!"}


# =====================================================
# INDEXAÇÃO / EXTRAÇÃO DE CHUNKS COM SALVAMENTO EM tmp/
# =====================================================

@app.post("/index/pdf")
async def index_pdf(file: UploadFile = File(...)):
    # Caminho do arquivo salvo localmente
    temp_path = os.path.join(UPLOAD_DIR, file.filename)

    # Salvar PDF enviado
    with open(temp_path, "wb") as f:
        f.write(await file.read())

    # Carregar e dividir PDF em chunks
    loader = PDFLoader()
    chunks = loader.load_chunks(temp_path)

    return {
        "filename": file.filename,
        "num_chunks": len(chunks),
        "chunks_sample": chunks[:3]  # Apenas 3 chunks para teste
    }


# =====================================================
# INGESTÃO COMPLETA: CHUNKS + EMBEDDINGS + FAISS + BM25
# =====================================================

@app.post("/ingest/pdf")
async def ingest_pdf(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="Arquivo ausente")

    # caminho do arquivo
    temp_path = os.path.join("tmp", file.filename)

    # salva PDF
    with open(temp_path, "wb") as f:
        f.write(await file.read())

    # extrai chunks
    chunks = loader.load_chunks(temp_path)

    if not chunks:
        return {"chunks_ingested": 0, "message": "Nenhum texto extraído do PDF"}

    texts = [c["text"] for c in chunks]

    # Ajustar metadados conforme seu loader atualizado
    metadata = [
        {
            "source": c["source"],
            "page": c["page"],
            "chunk_id": c["chunk_id"]
        }
        for c in chunks
    ]

    # gera embeddings
    vectors = embedder.encode(texts)

    if vectors.shape[1] != VECTOR_DIM:
        raise HTTPException(
            status_code=500,
            detail=f"Dimensão de embedding incompatível: {vectors.shape}"
        )

    # indexa FAISS
    vectorstore.add(vectors, texts, metadata=metadata)

    # indexa BM25
    bm25.add(texts)

    return {"chunks_ingested": len(texts)}


# -----------------------------
# BUSCAS
# -----------------------------

@app.get("/search")
async def search(q: str, k: int = 5):
    if not q:
        raise HTTPException(status_code=400, detail="Query vazia")

    q_vec = embedder.encode([q]).astype("float32")
    results = vectorstore.search(q_vec, k=k)

    return {"query": q, "results": results}


@app.get("/search/hybrid")
async def hybrid_search(q: str, k: int = 5):
    q_vec = embedder.encode([q]).astype("float32")
    sem_results = vectorstore.search(q_vec, k=k)
    lex_results = bm25.search(q, k=k)

    return {
        "query": q,
        "semantic": sem_results,
        "lexical": lex_results
    }


@app.post("/query")
async def query(
        q: str = Form(...),
        k: int = Form(5),
        rerank_k: int = Form(20)
):
    result = vector_service.semantic_search(q, k=k, rerank_k=rerank_k)
    return {"query": q, "results": result}

@app.post("/ask")
async def ask_question(payload: dict):
    query = payload["query"]

    # 1) BM25
    bm25_hits = bm25.search(query, k=10)

    # 2) FAISS (semantic)
    vect = embedder.embed([query])
    faiss_hits = faiss_store.search(vect, k=10)

    # 3) Reranking cruzado
    reranked = reranker.rerank(query, bm25_hits + faiss_hits)
    top_chunks = [x["text"] for x in reranked[:5]]

    # 4) Geração com BioMistral
    answer = generator.generate(query, top_chunks)

    return {"answer": answer}


# -----------------------------
# NER
# -----------------------------

@app.post("/ner")
async def run_ner(text: str = Form(...)):
    if not text:
        raise HTTPException(status_code=400, detail="Texto vazio")
    return ner.extract(text)


# -----------------------------
# Inicialização direta
# -----------------------------

def start_api():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
