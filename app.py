import os
from dotenv import load_dotenv
import streamlit as st
import difflib

try:
    import pysqlite3
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

# ─── 1. Chargement de la clé API ──────────────────────────────────────────────
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
if not GOOGLE_API_KEY:
    st.error("🚨 GOOGLE_API_KEY manquante dans .env")
    st.stop()
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# Configure google.generativeai en v2
import google.generativeai as genai
genai.configure(api_key=GOOGLE_API_KEY)

# ─── 2. Imports LangChain & Chroma ───────────────────────────────────────────
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ─── 3. Initialisation embeddings + ChromaDB ────────────────────────────────
embedding = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=GOOGLE_API_KEY
)
vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embedding)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# ─── 4. Initialisation du LLM ───────────────────────────────────────────────
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.5,
    google_api_key=GOOGLE_API_KEY
)

# ─── 5. Prompt template RAG ─────────────────────────────────────────────────
LLM_CONTEXT = """
Prompt:

You are a research question-answering assistant with access to a curated set of academic papers on student performance.

You must strictly follow these instructions:

Answer only based on the provided context. Never speculate or rely on external knowledge.

If the answer is not found in the context, respond exactly:

The answer is not available in the provided documents.

If the question is off-topic (not about student performance), respond exactly:

Your question is so fucking stupid, you piece of shit. I'm here only to answer questions about student performances.

Keep your answer concise, using a maximum of five sentences.

Format your answer in Markdown.

Respond in the same language as the question.

Example

Question:
Quels sont les facteurs les plus corrélés à la réussite des étudiants selon les études disponibles ?

Answer:
Plusieurs études indiquent que l’assiduité, le niveau socio-économique et la motivation intrinsèque sont des facteurs fortement corrélés à la réussite des étudiants. Par exemple, le document student_success_analysis.pdf met en avant l’impact positif d’une forte participation en classe.

Input:

Question: {question}
Context: {context}
Answer:
"""
llm_prompt = PromptTemplate.from_template(LLM_CONTEXT)

# ─── 6. Helpers pour RAG ────────────────────────────────────────────────────
def retrieve_and_format(query: str) -> str:
    """Récupère les chunks et formate le contexte brut pour le LLM."""
    results = vectorstore.similarity_search_with_score(query, k=5)
    # On concatène simplement les contenus pour le contexte
    return "\n\n".join(doc.page_content for doc, _ in results)

# ─── 7. Construction de la RAG chain ────────────────────────────────────────
rag_chain = (
    {"context": RunnableLambda(retrieve_and_format), "question": RunnablePassthrough()}
    | llm_prompt
    | llm
    | StrOutputParser()
)

# ─── 8. Helper pour l’affichage des sources uniques ───────────────────────
def get_pdf_sources_scores(question: str, top_k: int = 5) -> list[tuple[str, float]]:
    """
    Récupère les résultats vectoriels, puis retourne les Top_k
    de PDF uniques avec le meilleur score (distance) par PDF.
    """
    raw = vectorstore.similarity_search_with_score(question, k=top_k + 10)
    best_by_pdf: dict[str, float] = {}
    for doc, score in raw:
        src = doc.metadata.get("source", "N/A")
        # on garde le score minimal (distance) par PDF
        if src not in best_by_pdf or score < best_by_pdf[src]:
            best_by_pdf[src] = score

    # trier par score croissant et prendre top_k
    sorted_pdfs = sorted(best_by_pdf.items(), key=lambda x: x[1])[:top_k]
    return sorted_pdfs

# ─── 9. Interface Streamlit ────────────────────────────────────────────────
st.set_page_config(page_title="Student RAG Chat", page_icon="🎓")
st.title("🎓 Assistant de recherche académique")
st.markdown("Pose une question sur les performances des étudiants.")

if 'attempt_count' not in st.session_state:
    st.session_state.attempt_count = 0

question = st.text_input("💬 Votre question")

if question:
    # Génération de la réponse RAG
    answer = rag_chain.invoke(question)

    # Vérifier si la réponse est valide
    if "fucking" not in str(answer).lower() and "not available" not in str(answer).lower():
        st.session_state.attempt_count = 0  # Réinitialiser le compteur
        st.markdown("### 📘 Réponse")
        st.markdown(answer)

        # Affichage des noms de PDF et scores
        st.markdown("---")
        st.markdown("### 📚 Sources utilisées (PDF & Score)")
        for i, (pdf_name, score) in enumerate(get_pdf_sources_scores(question), 1):
            st.markdown(f"{i}. **{pdf_name}** — Score : {score:.4f}")
    else:
        st.session_state.attempt_count += 1
        if st.session_state.attempt_count < 5:
            st.warning("Essayez à nouveau : question hors sujet ou non couverte.")
        elif st.session_state.attempt_count < 10:
            st.error("Le sujet est la performance des étudiants et la question doit correspondre aux recherches disponibles. Soyez précis.")
        else:
            st.markdown("### 📘 Réponse")
            st.markdown(answer)
