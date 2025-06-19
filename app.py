import os
from dotenv import load_dotenv
import streamlit as st
import difflib
from datetime import datetime
from typing import List, Dict, Tuple
import json
import time
import random
import logging

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

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    top_p=0.9,
    max_retries=3,
    google_api_key=GOOGLE_API_KEY
)

# ─── 5. Rate Limited Retriever Class ─────────────────────────────────────────
class RateLimitedRetriever:
    """Wrapper around retriever with rate limiting and retry logic"""

    def __init__(self, retriever, max_retries=3, base_delay=1.0, max_delay=60.0):
        self.retriever = retriever
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.last_request_time = 0
        self.min_interval = 0.5  # Minimum seconds between requests

    def _wait_if_needed(self):
        """Ensure minimum interval between requests"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_interval:
            sleep_time = self.min_interval - elapsed
            logger.info(f"Rate limiting: waiting {sleep_time:.2f}s")
            time.sleep(sleep_time)

    def _exponential_backoff(self, attempt):
        """Calculate exponential backoff delay"""
        delay = min(self.base_delay * (2 ** attempt) + random.uniform(0, 1), self.max_delay)
        return delay

    def get_relevant_documents(self, query: str):
        """Get documents with rate limiting and retry logic"""
        self._wait_if_needed()

        for attempt in range(self.max_retries + 1):
            try:
                self.last_request_time = time.time()
                result = self.retriever.get_relevant_documents(query)
                logger.info(f"Successfully retrieved {len(result)} documents")
                return result

            except Exception as e:
                error_msg = str(e).lower()

                if "rate_limit_exceeded" in error_msg or "429" in error_msg or "quota" in error_msg:
                    if attempt < self.max_retries:
                        delay = self._exponential_backoff(attempt)
                        logger.warning(f"Rate limit hit. Attempt {attempt + 1}/{self.max_retries + 1}. Waiting {delay:.2f}s")
                        time.sleep(delay)
                        continue
                    else:
                        logger.error("Max retries exceeded for rate limiting")
                        raise Exception("Rate limit exceeded after all retries. Please wait before making more requests.")
                else:
                    # Non-rate-limit error, re-raise immediately
                    logger.error(f"Non-rate-limit error: {e}")
                    raise e

        return []

# ─── 6. Conversation Memory Class ───────────────────────────────────────────
class ConversationMemory:
    """Conversation memory manager for RAG"""

    def __init__(self, max_history=3, max_context_length=900):
        self.history: List[Dict] = []  # [{question, response, timestamp, sources}]
        self.max_history = max_history
        self.max_context_length = max_context_length

    def add_exchange(self, question: str, response: str, sources: List[str] = None):
        """Add an exchange to the history"""
        exchange = {
            'question': question,
            'response': response,
            'timestamp': datetime.now().isoformat(),
            'sources': sources or []
        }
        self.history.append(exchange)

        # Limit history size
        if len(self.history) > self.max_history:
            self.history.pop(0)

    def get_context_summary(self) -> str:
        """Generate a summary of recent history"""
        if not self.history:
            return "No previous conversation."

        context_parts = []
        for i, exchange in enumerate(self.history[-3:], 1):  # Last 3 exchanges
            context_parts.append(
                f"Exchange {i}:\n"
                f"Q: {exchange['question']}\n"
                f"A: {exchange['response'][:300]}..."  # First 300 chars of response
            )

        return "\n\n".join(context_parts)

# ─── 7. Contextual RAG Class ────────────────────────────────────────────────
class ContextualRAG:
    """RAG with conversational memory capabilities - always includes history"""

    def __init__(self, retriever, llm, memory: ConversationMemory):
        self.retriever = retriever
        self.llm = llm
        self.memory = memory

        # Updated prompt that always includes history and lets LLM decide
        self.LLM_CONTEXT = """
Conversation History:
{conversation_history}

You are a research question-answering assistant with access to a curated set of academic papers on student performance.
You are an expert on everything related to student performance, including grades, lifestyle, mental health, study habits, educational psychology, learning strategies, academic motivation, and student well-being.

The conversation history above is provided for context. Use it if the current question references previous exchanges, but ignore it if the question is completely independent.

INSTRUCTIONS (follow in order of priority):

1. **PRIMARY SOURCE**: Always prioritize information from the provided context/documents when available.

2. **SUPPLEMENTARY KNOWLEDGE**: If the question is about student performance or closely related topics (education, learning, academic success, student well-being, etc.) BUT the provided context lacks sufficient detail or doesn't fully address the question, you may supplement with your expert knowledge to provide a complete answer.

3. **INTEGRATION**: When using both sources, clearly indicate what comes from the documents vs. your additional expertise. Use phrases like:
   - "Based on the provided research..." (for document content)
   - "Additionally, research in this area suggests..." (for supplementary knowledge)

4. **UNAVAILABLE INFORMATION**: If the question is about student performance but neither the documents nor your knowledge can adequately answer it, respond exactly:
   "The answer is not available in the provided documents, and I don't have sufficient additional knowledge on this specific aspect."

5. **OFF-TOPIC QUESTIONS**: If the question is completely unrelated to student performance, education, or learning, respond exactly:
   "blablabla"

**FORMATTING REQUIREMENTS**:
- Keep your answer concise but comprehensive, using a maximum of seven sentences when combining sources
- Format your answer in Markdown
- Respond in the same language as the question
- When citing documents, mention the source when possible

Current Question: {question}
Context: {context}
Answer:
"""

        self.llm_prompt = PromptTemplate.from_template(self.LLM_CONTEXT)

    def format_docs_with_sources(self, docs):
        """Format documents with their sources"""
        return "\n\n".join([
            f"[Source: {doc.metadata.get('source', 'N/A')}]\n{doc.page_content}"
            for doc in docs
        ])

    def invoke(self, question: str) -> str:
        """Process a question with conversation history always included"""

        try:
            # 1. Retrieve relevant documents using the original question
            relevant_docs = self.retriever.get_relevant_documents(question)

            # 2. Prepare contexts
            document_context = self.format_docs_with_sources(relevant_docs)
            conversation_history = self.memory.get_context_summary()

            # 3. Generate response with history always included
            response = self.llm_prompt.invoke({
                "conversation_history": conversation_history,
                "question": question,
                "context": document_context
            })

            # 4. Get the actual response from LLM
            final_response = self.llm.invoke(response).content

            # 5. Extract sources used
            sources = [doc.metadata.get('source', 'N/A') for doc in relevant_docs[:3]]

            # 6. Update memory
            self.memory.add_exchange(question, final_response, sources)

            return final_response

        except Exception as e:
            error_msg = str(e)
            if "rate limit" in error_msg.lower() or "quota" in error_msg.lower():
                return "⚠️ **Rate limit reached**. Please wait a moment before asking another question."
            else:
                logger.error(f"Unexpected error: {e}")
                return f"❌ **Error occurred**: {error_msg}"

# ─── 8. Initialize new system ────────────────────────────────────────────────
# Create rate-limited retriever
rate_limited_retriever = RateLimitedRetriever(retriever)

# Initialize conversation memory and contextual RAG
if 'conversation_memory' not in st.session_state:
    st.session_state.conversation_memory = ConversationMemory(max_history=5)

if 'contextual_rag' not in st.session_state:
    st.session_state.contextual_rag = ContextualRAG(
        rate_limited_retriever, 
        llm, 
        st.session_state.conversation_memory
    )

# ─── 9. Helper functions ─────────────────────────────────────────────────────
def should_show_sources(response: str) -> bool:
    """Determine if sources should be shown based on the response content"""
    response_lower = response.lower().strip()

    # Don't show sources for these types of responses
    no_source_indicators = [
        "the answer is not available in the provided documents",
        "blablabla",
        "rate limit reached",
        "error occurred",
        "⚠️",
        "❌"
    ]

    # Check if response contains any of the non-answer indicators
    for indicator in no_source_indicators:
        if indicator in response_lower:
            return False

    return True

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

def process_question(question: str):
    """Process a question through the contextual RAG chain and return answer with sources."""
    answer = st.session_state.contextual_rag.invoke(question)
    
    # Only get sources if we should show them
    if should_show_sources(answer):
        sources = get_pdf_sources_scores(question)
    else:
        sources = []
    
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
st.markdown('<p style="text-align: center; color: #7F8C8D; font-size: 1.1rem; margin-bottom: 2rem;">Ask questions about student performance research with conversation memory</p>', unsafe_allow_html=True)

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
            # Clear conversation memory as well
            st.session_state.conversation_memory = ConversationMemory(max_history=5)
            st.session_state.contextual_rag = ContextualRAG(
                rate_limited_retriever, 
                llm, 
                st.session_state.conversation_memory
            )
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
    if "blablabla" not in str(answer).lower() and "not available" not in str(answer).lower():
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
    st.write(f"Conversation history entries: {len(st.session_state.conversation_memory.history)}")
    
    # Show conversation history
    st.markdown("### Conversation Memory")
    if st.session_state.conversation_memory.history:
        for i, exchange in enumerate(st.session_state.conversation_memory.history, 1):
            with st.expander(f"Exchange {i}"):
                st.write(f"**Q:** {exchange['question'][:100]}...")
                st.write(f"**A:** {exchange['response'][:100]}...")
                st.write(f"**Sources:** {', '.join(exchange['sources'])}")
    else:
        st.write("No conversation history yet")
    
    # Option pour activer/désactiver le few-shot prompting
    st.markdown("### System Status")
    st.write("❌ Few-Shot Prompting: Not active")
    st.write("✅ Conversation Memory: Active")
    st.write("✅ Rate Limiting: Active")
    st.write("✅ Contextual RAG: Active")
