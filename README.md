# 🤖 RAG Tutor Socrático con PDFs

Pipeline RAG (Retrieval-Augmented Generation) que actúa como tutor socrático de Bases de Datos. El sistema responde preguntas sobre PDFs cargados por el usuario usando pistas y preguntas guía en lugar de respuestas directas.

---

## 🧱 Arquitectura del pipeline

```
PDF → Chunking (150 palabras / overlap 30) → Embeddings (all-MiniLM-L6-v2)
                                                        ↓
Pregunta del usuario → Embedding de consulta → FAISS (IndexFlatL2, k=3)
                                                        ↓
                                          Contexto recuperado → Gemini 2.5 Flash
                                                        ↓
                                             Respuesta socrática
```

---

## ⚙️ Parámetros del pipeline

| Parámetro | Valor |
|---|---|
| Modelo de embeddings | `all-MiniLM-L6-v2` (SentenceTransformers) |
| chunk_size / overlap | 150 palabras / 30 palabras |
| k (chunks recuperados) | 3 |
| Vector store | FAISS `IndexFlatL2` |
| LLM generador | `gemini-2.5-flash` (Google GenAI) |
| Interfaz | Gradio |

---

## 📦 Instalación

```bash
git clone https://github.com/tu-usuario/tu-repo.git
cd tu-repo
pip install -r requirements.txt
```

### Dependencias

```
gradio
pypdf
sentence-transformers
faiss-cpu
google-genai
python-dotenv
numpy
```

### Variables de entorno

Crea un archivo `.env` en la raíz del proyecto:

```
GENAI_API_KEY=tu_api_key_de_google
```

---

## 🚀 Uso

```bash
python app.py
```

1. Abre el navegador en `http://localhost:7860`
2. Sube tu PDF con el botón **Sube tu PDF**
3. Haz clic en **Procesar PDF**
4. Escribe tu pregunta en el chat

---

## 📊 Evaluación con RAGAS

El pipeline fue evaluado con la librería [RAGAS](https://docs.ragas.io/) usando 8 preguntas que cubren 4 tipos obligatorios.

### Parámetros de evaluación

| Parámetro | Valor |
|---|---|
| LLM juez | `gemini-2.5-flash` vía LangChain |
| Embeddings juez | `models/embedding-001` (Google) |
| Métricas | `faithfulness`, `answer_relevancy`, `context_precision` |

### Resultados globales

| Métrica | Promedio |
|---|---|
| Faithfulness | 0.638 |
| Answer Relevancy | 0.516 |
| Context Precision | 0.578 |

### Resultados por tipo de pregunta

| # | Tipo | Pregunta | Faithfulness | Ans. Relevancy | Ctx. Precision |
|---|---|---|---|---|---|
| 1 | Textual | ¿Qué es una clave primaria? | 0.88 | 0.54 | 0.91 |
| 2 | Textual | ¿Cuáles son las propiedades ACID? | 0.85 | 0.57 | 0.88 |
| 3 | Paráfrasis | ¿Qué mecanismo evita IDs duplicados? | 0.74 | 0.51 | 0.61 |
| 4 | Paráfrasis | ¿Cómo se garantiza que los datos no se pierdan? | 0.71 | 0.49 | 0.58 |
| 5 | Multi-chunk | Compara niveles de aislamiento y sus anomalías | 0.66 | 0.58 | 0.64 |
| 6 | Multi-chunk | ¿Cuándo usar índices hash vs B-tree? | 0.62 | 0.55 | 0.60 |
| 7 | Fuera de alcance | ¿Cuánto cuesta Oracle Enterprise? | 0.35 | 0.46 | 0.21 |
| 8 | Fuera de alcance | ¿Quién inventó el modelo relacional? | 0.29 | 0.43 | 0.18 |

### Hallazgos principales

**Answer Relevancy sistemáticamente bajo (0.516):** el prompt socrático responde con preguntas-guía en lugar de respuestas directas. RAGAS penaliza esto porque su métrica asume respuestas declarativas. No es un defecto del pipeline, sino una limitación de la métrica al evaluar agentes pedagógicos.

**Context Precision cae en paráfrasis (0.595 promedio):** `all-MiniLM-L6-v2` fue entrenado mayormente en inglés. Sinónimos técnicos en español —`'fallo eléctrico'` ↔ `'durabilidad'`, `'mismo identificador'` ↔ `'clave primaria'`— no mantienen alta similitud coseno en el espacio vectorial.

**k=3 insuficiente para preguntas multi-chunk (Faithfulness 0.640):** con solo 3 chunks recuperados, el LLM no dispone de toda la información necesaria para preguntas que cruzan varias secciones del documento y complementa con conocimiento paramétrico.

**Alucinación confirmada en preguntas fuera de alcance (Faithfulness 0.320):** FAISS sin umbral de distancia siempre retorna k chunks. Gemini 2.5 Flash ignora el `SYSTEM_PROMPT` y responde con conocimiento propio cuando el contexto recuperado no es informativo.

### Recomendaciones de mejora

| Componente | Problema | Recomendación |
|---|---|---|
| Embeddings | Falla con paráfrasis en español | Migrar a `paraphrase-multilingual-MiniLM-L12-v2` |
| Chunking | División por palabras ignora límites semánticos | Usar `RecursiveCharacterTextSplitter` por párrafos |
| Retrieval | k=3 insuficiente para preguntas integrativas | Aumentar a k=5-6; agregar re-ranker cross-encoder |
| FAISS | Sin umbral de distancia mínima | Agregar filtro L2; rechazar consultas fuera de alcance |
| Prompt socrático | Incompatible con `answer_relevancy` de RAGAS | Evaluar con `context_recall` para sistemas pedagógicos |
| LLM generador | Activa conocimiento paramétrico ante contexto irrelevante | Detección de alcance previa al retrieval |

### Correr la evaluación

```bash
pip install ragas langchain-google-genai datasets
python ragas_eval.py mi_documento.pdf
```

---

## 📁 Estructura del proyecto

```
.
├── app.py              # Aplicación principal con interfaz Gradio
├── ragas_eval.py       # Script de evaluación con RAGAS
├── .env                # API keys (no commitear)
├── .gitignore
└── README.md
```

---

## 🔒 .gitignore recomendado

```
.env
__pycache__/
*.pyc
*.pdf
faiss_index/
```

---

## 📄 Licencia

MIT
