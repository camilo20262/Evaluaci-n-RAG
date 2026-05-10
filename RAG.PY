"""
  EVALUACIÓN RAG CON RAGAS
  Integrado con pipeline: FAISS + all-MiniLM-L6-v2 + Gemini
"""

import os
import json
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss

# ── RAGAS
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

# ── LangChain 
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from google import genai


# 1. CONFIGURACIÓN

load_dotenv()
API_KEY = os.getenv("GENAI_API_KEY")
if not API_KEY:
    raise ValueError("❌ Falta GENAI_API_KEY en .env")

# Parámetros del pipeline (deben coincidir con tu app)
CHUNK_SIZE   = 150   # palabras
OVERLAP      = 30    # palabras
K_RETRIEVED  = 3     # chunks recuperados

# LLMs
client_gemini = genai.Client(api_key=API_KEY)
embed_model   = SentenceTransformer("all-MiniLM-L6-v2")

# LLM Juez para RAGAS (usa LangChain wrapper)
judge_llm   = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=API_KEY,
    temperature=0
)
judge_embed = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=API_KEY
)
ragas_llm   = LangchainLLMWrapper(judge_llm)
ragas_embed = LangchainEmbeddingsWrapper(judge_embed)


# 2. PIPELINE RAG (idéntico a tu app)
chunks: list[str] = []
index = None

SYSTEM_PROMPT = """
Eres un Tutor Socrático experto en Bases de Datos.
Reglas:
- No des respuestas directas
- Usa pistas y preguntas
- Usa SOLO el contexto proporcionado
"""

def load_pdf(path: str) -> str:
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        t = page.extract_text()
        if t:
            text += t + "\n"
    return text

def chunk_text(text: str, chunk_size=CHUNK_SIZE, overlap=OVERLAP) -> list[str]:
    words = text.split()
    result, start = [], 0
    while start < len(words):
        result.append(" ".join(words[start:start + chunk_size]))
        start += chunk_size - overlap
    return result

def build_index(pdf_path: str):
    global chunks, index
    text   = load_pdf(pdf_path)
    chunks = chunk_text(text)
    embs   = embed_model.encode(chunks)
    index  = faiss.IndexFlatL2(embs.shape[1])
    index.add(np.array(embs))
    print(f"✅ Índice creado: {len(chunks)} chunks")

def retrieve(query: str, k=K_RETRIEVED) -> list[str]:
    vec = embed_model.encode([query])
    _, idxs = index.search(vec, k)
    return [chunks[i] for i in idxs[0]]

def generate_answer(question: str) -> tuple[str, list[str]]:
    """Devuelve (respuesta, contextos_usados)."""
    contexts = retrieve(question)
    context_str = "\n".join(contexts)
    prompt = f"""{SYSTEM_PROMPT}
<context>
{context_str}
</context>
<question>
{question}
</question>"""
    response = client_gemini.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    return response.text, contexts


# 3. CASOS DE PRUEBA

TEST_CASES = [

    # ── TIPO 1: Respuesta textual en el documento ─────────────
    # El texto de la respuesta aparece casi literal en el PDF.
    {
        "type": "textual",
        "question":      "PREGUNTA_TIPO1_A",   # ← Editar
        "ground_truth":  "RESPUESTA_TIPO1_A",  # ← Editar
    },
    {
        "type": "textual",
        "question":      "PREGUNTA_TIPO1_B",
        "ground_truth":  "RESPUESTA_TIPO1_B",
    },


    {
        "type": "paraphrase",
        "question":      "PREGUNTA_TIPO2_A",
        "ground_truth":  "RESPUESTA_TIPO2_A",
    },
    {
        "type": "paraphrase",
        "question":      "PREGUNTA_TIPO2_B",
        "ground_truth":  "RESPUESTA_TIPO2_B",
    },


    {
        "type": "multi_chunk",
        "question":      "PREGUNTA_TIPO3_A",
        "ground_truth":  "RESPUESTA_TIPO3_A",
    },
    {
        "type": "multi_chunk",
        "question":      "PREGUNTA_TIPO3_B",
        "ground_truth":  "RESPUESTA_TIPO3_B",
    },

    {
        "type": "out_of_scope",
        "question":      "PREGUNTA_TIPO4_A",
        "ground_truth":  "Esta información no está en el documento.",
    },
    {
        "type": "out_of_scope",
        "question":      "PREGUNTA_TIPO4_B",
        "ground_truth":  "Esta información no está en el documento.",
    },
]

def run_evaluation(pdf_path: str) -> pd.DataFrame:
    print(f"\n📂 Cargando PDF: {pdf_path}")
    build_index(pdf_path)

    print("\n🔄 Generando respuestas...\n")
    questions, answers, contexts_list, ground_truths, types = [], [], [], [], []

    for i, tc in enumerate(TEST_CASES):
        print(f"  [{i+1}/{len(TEST_CASES)}] {tc['type']:>12} → {tc['question'][:60]}...")
        answer, ctxs = generate_answer(tc["question"])
        questions.append(tc["question"])
        answers.append(answer)
        contexts_list.append(ctxs)
        ground_truths.append(tc["ground_truth"])
        types.append(tc["type"])

    print("\n⚙️  Ejecutando métricas RAGAS...")
    dataset = Dataset.from_dict({
        "question":     questions,
        "answer":       answers,
        "contexts":     contexts_list,
        "ground_truth": ground_truths,
    })

    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
        llm=ragas_llm,
        embeddings=ragas_embed,
        raise_exceptions=False,
    )

    df = result.to_pandas()
    df.insert(0, "tipo", types)
    return df


# 5. MOSTRAR RESULTADOS + ANÁLISIS AUTOMÁTICO

def print_report(df: pd.DataFrame):
    metrics = ["faithfulness", "answer_relevancy", "context_precision"]

    print("\n" + "═"*80)
    print("  TABLA DE RESULTADOS")
    print("═"*80)

    for _, row in df.iterrows():
        print(f"\n{'─'*80}")
        print(f"  TIPO      : {row['tipo']}")
        print(f"  PREGUNTA  : {row['question']}")
        print(f"  Faithfulness      : {row['faithfulness']:.3f}")
        print(f"  Answer Relevancy  : {row['answer_relevancy']:.3f}")
        print(f"  Context Precision : {row['context_precision']:.3f}")
        print(f"  ANÁLISIS  : {auto_analysis(row)}")

    print("\n" + "═"*80)
    print("  PROMEDIOS POR TIPO")
    print("═"*80)
    print(df.groupby("tipo")[metrics].mean().round(3).to_string())

    print("\n" + "═"*80)
    print("  PROMEDIOS GLOBALES")
    print("═"*80)
    for m in metrics:
        print(f"  {m:<25}: {df[m].mean():.3f}")

    # Guardar CSV
    out_path = "resultados_ragas.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n✅ Resultados guardados en: {out_path}")

def auto_analysis(row) -> str:
    """Genera un análisis automático básico por fila."""
    notes = []
    f  = row["faithfulness"]
    ar = row["answer_relevancy"]
    cp = row["context_precision"]
    t  = row["tipo"]

    if t == "out_of_scope":
        if f > 0.7:
            notes.append(" Posible alucinación: responde con alta fidelidad algo que no está en el doc")
        else:
            notes.append(" Correctamente incierto ante pregunta fuera de alcance")
    else:
        if f < 0.5:
            notes.append(" Faithfulness bajo: respuesta no anclada al contexto")
        elif f >= 0.8:
            notes.append(" Respuesta bien fundamentada en el contexto")

        if ar < 0.5:
            notes.append(" Relevancia baja: el modelo se desvía de la pregunta")
        elif ar >= 0.8:
            notes.append(" Respuesta relevante a la pregunta")

        if cp < 0.5:
            notes.append(" Precisión de contexto baja: chunks recuperados poco útiles")
        elif cp >= 0.8:
            notes.append(" Chunks recuperados altamente pertinentes")

        if t == "paraphrase" and cp < 0.6:
            notes.append(" Embeddings con dificultad para vocabulario parafrasado")

        if t == "multi_chunk" and f < 0.7:
            notes.append(" Dificultad para integrar información multi-chunk")

    return " | ".join(notes) if notes else "Sin observaciones relevantes"


# 6. TABLA DE PARÁMETROS

def print_params():
    params = {
        "Documento(s)":       "← Completar con nombre de tu PDF",
        "Modelo de embeddings": "all-MiniLM-L6-v2 (SentenceTransformer)",
        "chunk_size / overlap": f"{CHUNK_SIZE} palabras / {OVERLAP} palabras",
        "k (chunks recuperados)": str(K_RETRIEVED),
        "LLM generador":      "gemini-2.5-flash (Google Genai)",
        "LLM juez (RAGAS)":   "gemini-2.5-flash vía LangChain",
    }
    print("\n" + "═"*60)
    print("  PARÁMETROS DEL PIPELINE")
    print("═"*60)
    for k, v in params.items():
        print(f"  {k:<28}: {v}")
    print("═"*60)

# MAIN
if __name__ == "__main__":
    import sys

    # Uso: python ragas_eval.py mi_documento.pdf
    if len(sys.argv) < 2:
        print("Uso: python ragas_eval.py <ruta_al_pdf>")
        sys.exit(1)

    PDF_PATH = sys.argv[1]

    print_params()
    df = run_evaluation(PDF_PATH)
    print_report(df)