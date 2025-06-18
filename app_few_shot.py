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

# ─── 1. Loading API key ──────────────────────────────────────────────
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
if not GOOGLE_API_KEY:
    st.error("GOOGLE_API_KEY missing in .env")
    st.stop()
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# Configure google.generativeai en v2
import google.generativeai as genai
genai.configure(api_key=GOOGLE_API_KEY)

# ─── 2. LangChain & Chroma imports ───────────────────────────────────────────
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate, FewShotPromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ─── 3. Embeddings + ChromaDB initialization ────────────────────────────────
embedding = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=GOOGLE_API_KEY
)
vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embedding)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# ─── 4. LLM initialization ───────────────────────────────────────────────
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.5,
    google_api_key=GOOGLE_API_KEY
)

# ─── 5. Définir les exemples pour le few-shot prompting ─────────────────────────────────────────────────
examples = [
    {
        "question": "How can learning analytics be used to enhance student performance?",
        "answer": """
Learning analytics can be used to enhance student performance by combining data from various sources such as student characteristics, prior academic history, and interactions with online and offline resources. For example, a study showed that using diverse data sources can improve the accuracy of predictive models by 20% (Smith et al., 2020).
Source: learning_analytics_study.pdf
"""
    },
    {
        "question": "What are the benefits of analyzing student digital footprints?",
        "answer": """
Analyzing student digital footprints can provide valuable insights into student behavior and engagement. By leveraging data from platforms managing student registration and Learning Management Systems, educators can gather clickstream data to improve student engagement by 30% (Johnson, 2019).
Source: digital_footprints_analysis.pdf
"""
    },
    {
        "question": "How can predictive models help identify students in need of assistance?",
        "answer": """
Predictive models can help identify students in need of assistance by using student characteristics, academic history, and interactions with online resources. For instance, a predictive model was able to identify at-risk students with 85% accuracy (Brown, 2021).
Source: predictive_models_study.pdf
"""
    },
    {
        "question": "What is adaptive feedback and how does it benefit students?",
        "answer": """
Adaptive feedback is personalized feedback based on each student's progression, providing guidance when needed. Studies have shown that students receiving adaptive feedback improved their performance by an average of 15% (Lee, 2022).
Source: adaptive_feedback_study.pdf
"""
    },
    {
        "question": "How can monitoring student activities improve their performance?",
        "answer": """
Monitoring student activities through various methods such as Group Scribbles and MTClassroom can help track student engagement and performance. For example, using these tools reduced the failure rate by 10% in some institutions (Williams, 2023).
Source: student_monitoring_study.pdf
"""
    },
    {
        "question": "As a teacher, how can I use learning analytics to collect data for tracking, monitoring, and enhancing students’ performance?",
        "answer": """
As a teacher, you can use learning analytics to collect data for tracking, monitoring, and enhancing students’ performance by:

* Using diverse data sources: Combine data from various sources such as student characteristics, prior academic history, programming laboratory work, and logged interactions with online and offline resources. This approach can improve the accuracy of predictive models by 20% (Smith et al., 2020).

* Analyzing student digital footprints: Leverage data from platforms managing student registration, custom learning platforms for programming submissions, and Learning Management Systems to gather clickstream data. This method can improve student engagement by 30% (Johnson, 2019).

* Implementing predictive models: Build predictive models using student characteristics, academic history, programming lab work, and interactions with online resources to identify students who need assistance. Predictive models have been shown to identify at-risk students with 85% accuracy (Brown, 2021).

* Providing adaptive feedback: Generate personalized feedback based on each student’s progression and provide guidance when needed. Adaptive feedback has been shown to improve student performance by an average of 15% (Lee, 2022).

* Monitoring student activities: Track student activities through various methods such as Group Scribbles, MTClassroom, and the Formative Assessment with Computational Technology (FACT) system. These tools have been shown to reduce the failure rate by 10% in some institutions (Williams, 2023).

Sources: learning_analytics_study.pdf, digital_footprints_analysis.pdf, predictive_models_study.pdf, adaptive_feedback_study.pdf, student_monitoring_study.pdf
"""
    }
]

# Template pour formater les exemples
example_prompt = PromptTemplate(
    input_variables=["question", "answer"],
    template="Question: {question}\nAnswer: {answer}"
)

# Création du FewShotPromptTemplate
few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    suffix="Question: {question}\nContext: {context}\nAnswer:",
    prefix="Here are some examples of questions and answers about student performance:",
    input_variables=["question", "context"]
)

# ─── 6. RAG helpers ────────────────────────────────────────────────────
def retrieve_and_format(query: str) -> str:
    """Retrieves chunks and formats raw context for the LLM."""
    results = vectorstore.similarity_search_with_score(query, k=5)
    # We simply concatenate the contents for the context
    return "\n\n".join(doc.page_content for doc, _ in results)

# ─── 7. RAG chain construction ────────────────────────────────────────────
rag_chain = (
    {"context": RunnableLambda(retrieve_and_format), "question": RunnablePassthrough()}
    | few_shot_prompt
    | llm
    | StrOutputParser()
)

# ─── 8. Helper for displaying unique sources ───────────────────────────────
def get_pdf_sources_scores(question: str, top_k: int = 5) -> list[tuple[str, float]]:
    """
    Retrieves vector results, then returns the Top_k
    unique PDFs with the best score (distance) per PDF.
    """
    raw = vectorstore.similarity_search_with_score(question, k=top_k + 10)
    best_by_pdf: dict[str, float] = {}
    for doc, score in raw:
        src = doc.metadata.get("source", "N/A")
        # we keep the minimal score (distance) per PDF
        if src not in best_by_pdf or score < best_by_pdf[src]:
            best_by_pdf[src] = score
    # sort by ascending score and take top_k
    sorted_pdfs = sorted(best_by_pdf.items(), key=lambda x: x[1])[:top_k]
    return sorted_pdfs

# ─── 9. RAG processing function ────────────────────────────────────────────
def process_question(question: str):
    """Process a question through the RAG chain and return answer with sources."""
    answer = rag_chain.invoke(question)
    sources = get_pdf_sources_scores(question)
    return answer, sources

# ─── 10. Custom CSS Styling ────────────────────────────────────────────────
def load_css():
    st.markdown("""
    <style>
    /* Main app styling - warm beige theme */
    .stApp {
        background-color: #FAF7F2;
        color: #2C3E50;
    }

    /* Hide streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Custom title styling - simple elegant */
    .main-title {
        text-align: center;
        padding: 2rem 0 1rem 0;
        font-size: 2.5rem;
        font-weight: 600;
        color: #34495E;
        border-bottom: 2px solid #E8E4DD;
        margin-bottom: 2rem;
    }

    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    */

    /* Chat container */
    .chat-container {
        max-width: 800px;
        margin: 0 auto;
        padding: 0 1rem;
    }

    /* User message bubble - right aligned */
    .user-message {
        display: flex;
        justify-content: flex-end;
        margin: 1rem 0;
        animation: fadeInRight 0.3s ease-in;
    }

    .user-bubble {
        background: linear-gradient(135deg, #E3F2FD 0%, #BBDEFB 100%);
        color: #1565C0;
        padding: 12px 18px;
        border-radius: 20px 20px 5px 20px;
        max-width: 70%;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        position: relative;
    }

    /* Assistant message bubble - left aligned */
    .assistant-message {
        display: flex;
        justify-content: flex-start;
        margin: 1rem 0;
        animation: fadeInLeft 0.3s ease-in;
    }

    .assistant-bubble {
        background: #F8F9FA;
        color: #2C3E50;
        padding: 12px 18px;
        border-radius: 20px 20px 20px 5px;
        max-width: 70%;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        border: 1px solid #E9ECEF;
    }

    /* Avatar styling */
    .message-avatar {
        width: 35px;
        height: 35px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 18px;
        margin: 0 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }

    .user-avatar {
        background: linear-gradient(135deg, #6B73FF, #9D50BB);
        color: white;
    }

    .assistant-avatar {
        background: linear-gradient(135deg, #A8EDEA, #FED6E3);
        color: #2C3E50;
    }

    /* Animations */
    @keyframes fadeInRight {
        from { opacity: 0; transform: translateX(20px); }
        to { opacity: 1; transform: translateX(0); }
    }

    @keyframes fadeInLeft {
        from { opacity: 0; transform: translateX(-20px); }
        to { opacity: 1; transform: translateX(0); }
    }

    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #6B73FF, #9D50BB);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 1.5rem;
        font-weight: 500;
        transition: all 0.3s ease;
        box-shadow: 0 2px 10px rgba(107, 115, 255, 0.3);
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(107, 115, 255, 0.4);
    }

    /* Sources styling */
    .sources-container {
        background: #F1F3F4;
        border-radius: 10px;
        padding: 15px;
        margin-top: 10px;
        border-left: 4px solid #6B73FF;
    }

    /* Input styling */
    .stTextInput > div > div > input {
        border-radius: 25px;
        border: 2px solid #E8E4DD;
        padding: 12px 20px;
        font-size: 16px;
        transition: all 0.3s ease;
    }

    .stTextInput > div > div > input:focus {
        border-color: #6B73FF;
        box-shadow: 0 0 0 3px rgba(107, 115, 255, 0.1);
    }

    /* Warning and error styling */
    .stAlert {
        border-radius: 15px;
        border: none;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }

    /* Sidebar styling */
    .css-1d391kg {
        background: #F1F3F4;
        border-right: 2px solid #E8E4DD;
    }

    </style>
    """, unsafe_allow_html=True)

# ─── 11. Custom message display functions ──────────────────────────────────
def display_user_message(content):
    st.markdown(f"""
    <div class="user-message">
        <!-- <div class="user-avatar">👤</div> -->
        <div class="user-bubble">{content}</div>
    </div>
    """, unsafe_allow_html=True)

def display_assistant_message(content, sources=None):
    st.markdown(f"""
    <div class="assistant-message">
        <!-- <div class="assistant-avatar">🤖</div> -->
        <div class="assistant-bubble">{content}</div>
    </div>
    """, unsafe_allow_html=True)

    if sources:
        with st.expander("📚 Sources used (PDF & Score)", expanded=False):
            sources_html = ""
            for i, (pdf_name, score) in enumerate(sources, 1):
                sources_html += f"<p><strong>{i}. {pdf_name}</strong> — Score: {score:.4f}</p>"
            st.markdown(f'<div class="sources-container">{sources_html}</div>', unsafe_allow_html=True)

# ─── 12. Streamlit interface ────────────────────────────────────────────────
st.set_page_config(
    page_title="Student Performance RAG Chat",
    page_icon="🎓",
    layout="wide"
)

# Load custom CSS
load_css()

# Custom title
st.markdown('<h1 class="main-title">Learning Analytics Predictive Assistant</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #7F8C8D; font-size: 1.1rem; margin-bottom: 2rem;">Ask questions about student performance research</p>', unsafe_allow_html=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'attempt_count' not in st.session_state:
    st.session_state.attempt_count = 0

# Chat interface with container
with st.container():
    # Clear chat button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.attempt_count = 0
            st.rerun()

    # Chat messages container
    chat_container = st.container()
    with chat_container:
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)

        # Display chat history with custom styling
        for message in st.session_state.messages:
            if message["role"] == "user":
                display_user_message(message["content"])
            else:
                sources = message.get("sources", None)
                display_assistant_message(message["content"], sources)

        st.markdown('</div>', unsafe_allow_html=True)

# Chat input
if prompt := st.chat_input("Type your question here..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message
    display_user_message(prompt)

    # Process the question
    with st.spinner("Thinking..."):
        answer, sources = process_question(prompt)

    # Check if the response is valid
    if "fucking" not in str(answer).lower() and "not available" not in str(answer).lower():
        # Valid response - display it
        display_assistant_message(answer, sources)

        # Add to chat history
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources
        })

        # Reset attempt counter AFTER successful response
        st.session_state.attempt_count = 0

    else:
        # Invalid response - handle attempt count
        st.session_state.attempt_count += 1

        if st.session_state.attempt_count < 5:
            warning_msg = "Try again: off-topic question or not covered."
            display_assistant_message(warning_msg)
            st.session_state.messages.append({
                "role": "assistant",
                "content": warning_msg
            })
        elif st.session_state.attempt_count < 10:
            error_msg = "The topic is about student performance and the question must match available research. Please be specific."
            display_assistant_message(error_msg)
            st.session_state.messages.append({
                "role": "assistant",
                "content": error_msg
            })
        else:
            # Easter egg - show the raw answer
            display_assistant_message(answer)
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer
            })

# Debug sidebar
with st.sidebar:
    st.markdown("### Debug Info")
    st.write(f"Attempt count: {st.session_state.attempt_count}")
    st.write(f"Total messages: {len(st.session_state.messages)}")
