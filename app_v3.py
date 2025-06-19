import os
from dotenv import load_dotenv
import streamlit as st
import difflib
from datetime import datetime
from typing import List, Dict, Optional

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

# ─── 3. Conversation Memory Class ───────────────────────────────────────────
class ConversationMemory:
    """Lightweight conversation memory for Streamlit"""
    
    def __init__(self, max_history=5):
        self.max_history = max_history
    
    def get_recent_context(self, messages: List[Dict], max_exchanges=3) -> str:
        """Generate context from recent exchanges"""
        if not messages or len(messages) < 2:
            return ""
        
        # Get last exchanges (user-assistant pairs)
        context_parts = []
        exchanges = []
        
        # Group messages into exchanges
        for i in range(0, len(messages), 2):
            if i + 1 < len(messages):
                user_msg = messages[i]
                assistant_msg = messages[i + 1]
                if user_msg["role"] == "user" and assistant_msg["role"] == "assistant":
                    exchanges.append((user_msg["content"], assistant_msg["content"]))
        
        # Take last few exchanges
        recent_exchanges = exchanges[-max_exchanges:] if len(exchanges) > max_exchanges else exchanges
        
        for i, (question, answer) in enumerate(recent_exchanges, 1):
            context_parts.append(
                f"Previous Exchange {i}:\n"
                f"Question: {question}\n"
                f"Answer: {answer[:150]}..."
            )
        
        return "\n\n".join(context_parts)
    
    def is_contextual_question(self, question: str) -> bool:
        """Check if question needs conversation context"""
        contextual_indicators = [
            'and in', 'and for', 'also', 'too', 'as well', 'what about', 'how about',
            'this', 'that', 'these', 'those', 'it', 'they', 'them',
            'same thing', 'likewise', 'similar', 'comparable',
            'et à', 'et pour', 'aussi', 'également', 'pareillement',
            'qu\'en est-il de', 'quid de', 'concernant', 'à propos de',
            'cette', 'ce', 'cela', 'ça', 'il', 'elle', 'ils', 'elles',
            'la même chose', 'idem', 'similaire', 'comparable'
        ]
        
        question_lower = question.lower()
        return any(indicator in question_lower for indicator in contextual_indicators)
    
    def reformulate_question(self, question: str, messages: List[Dict], llm) -> str:
        """Reformulate contextual questions to be standalone"""
        if not self.is_contextual_question(question) or not messages:
            return question
        
        context = self.get_recent_context(messages, max_exchanges=2)
        if not context:
            return question
        
        reformulation_prompt = f"""
Conversation History:
{context}

Current Question: {question}

If the current question refers to the conversation history (uses pronouns, implicit references, etc.),
reformulate it to make it standalone and clear. Otherwise, return the question as is.

Reformulated Question:
"""
        
        try:
            reformulated = llm.invoke(reformulation_prompt).content
            return reformulated.strip()
        except:
            return question

# ─── 4. Initialize conversation memory ───────────────────────────────────────
@st.cache_resource
def get_conversation_memory():
    return ConversationMemory(max_history=5)

# ─── 5. Embeddings + ChromaDB initialization ────────────────────────────────
embedding = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=GOOGLE_API_KEY
)
vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embedding)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# ─── 6. LLM initialization ───────────────────────────────────────────────
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.5,
    google_api_key=GOOGLE_API_KEY
)

# ─── 7. Enhanced examples with conversation context ─────────────────────────
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
        "question": "As a teacher, how can I use learning analytics to collect data for tracking, monitoring, and enhancing students' performance?",
        "answer": """
As a teacher, you can use learning analytics to collect data for tracking, monitoring, and enhancing students' performance by:

* Using diverse data sources: Combine data from various sources such as student characteristics, prior academic history, programming laboratory work, and logged interactions with online and offline resources. This approach can improve the accuracy of predictive models by 20% (Smith et al., 2020).

* Analyzing student digital footprints: Leverage data from platforms managing student registration, custom learning platforms for programming submissions, and Learning Management Systems to gather clickstream data. This method can improve student engagement by 30% (Johnson, 2019).

* Implementing predictive models: Build predictive models using student characteristics, academic history, programming lab work, and interactions with online resources to identify students who need assistance. Predictive models have been shown to identify at-risk students with 85% accuracy (Brown, 2021).

* Providing adaptive feedback: Generate personalized feedback based on each student's progression and provide guidance when needed. Adaptive feedback has been shown to improve student performance by an average of 15% (Lee, 2022).

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

# ─── 8. Enhanced prompt with conversation context ───────────────────────────
def create_contextual_prompt():
    """Create prompt template with conversation context"""
    return FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        prefix="""
Conversation History (for context only):
{conversation_history}

You are a research question-answering assistant with access to a curated set of academic papers on student performance.

You must strictly follow these instructions:

Answer only based on the provided context. Never speculate or rely on external knowledge.

If the answer is not found in the context, respond exactly:

The answer is not available in the provided documents.

If the question is off-topic (not about student performance), respond exactly:

Your questions and prompts are so fucking stupid, you piece of shit. I'm here only to answer questions about student performances.

Keep your answer concise, using a maximum of five sentences.

Format your answer in Markdown.

Respond in the same language as the question.

Consider the conversation history above when interpreting contextual references in the current question.

Here are some examples of questions and answers about student performance:
""",
        suffix="Question: {question}\nContext: {context}\nAnswer:",
        input_variables=["question", "context", "conversation_history"]
    )

# ─── 9. Enhanced RAG helpers ────────────────────────────────────────────────
def retrieve_and_format_contextual(query: str, messages: List[Dict], memory: ConversationMemory) -> str:
    """Enhanced retrieval with conversation context"""
    # Reformulate question if needed
    reformulated_query = memory.reformulate_question(query, messages, llm)
    
    # Use reformulated query for better retrieval
    search_query = reformulated_query if reformulated_query != query else query
    results = vectorstore.similarity_search_with_score(search_query, k=5)
    
    return "\n\n".join(doc.page_content for doc, _ in results)

def get_conversation_context(messages: List[Dict], memory: ConversationMemory) -> str:
    """Get formatted conversation context"""
    return memory.get_recent_context(messages, max_exchanges=3)

# ─── 10. Enhanced RAG chain construction ────────────────────────────────────
def create_contextual_rag_chain(memory: ConversationMemory):
    """Create RAG chain with conversation context"""
    contextual_prompt = create_contextual_prompt()
    
    def enhanced_retrieve_and_format(inputs):
        query = inputs["question"]
        messages = inputs.get("messages", [])
        
        # Get conversation context
        conversation_history = get_conversation_context(messages, memory)
        
        # Enhanced retrieval
        context = retrieve_and_format_contextual(query, messages, memory)
        
        return {
            "question": query,
            "context": context,
            "conversation_history": conversation_history
        }
    
    return (
        RunnableLambda(enhanced_retrieve_and_format)
        | contextual_prompt
        | llm
        | StrOutputParser()
    )

# ─── 11. Helper for displaying unique sources ───────────────────────────────
def get_pdf_sources_scores_contextual(question: str, messages: List[Dict], memory: ConversationMemory, top_k: int = 5) -> list[tuple[str, float]]:
    """Enhanced source retrieval with context"""
    # Use reformulated question for better source matching
    reformulated_question = memory.reformulate_question(question, messages, llm)
    search_query = reformulated_question if reformulated_question != question else question
    
    raw = vectorstore.similarity_search_with_score(search_query, k=top_k + 10)
    best_by_pdf: dict[str, float] = {}
    
    for doc, score in raw:
        src = doc.metadata.get("source", "N/A")
        if src not in best_by_pdf or score < best_by_pdf[src]:
            best_by_pdf[src] = score

    sorted_pdfs = sorted(best_by_pdf.items(), key=lambda x: x[1])[:top_k]
    return sorted_pdfs

# ─── 12. Enhanced processing function ────────────────────────────────────────
def process_question_contextual(question: str, messages: List[Dict], memory: ConversationMemory):
    """Enhanced question processing with conversation context"""
    # Create contextual RAG chain
    rag_chain = create_contextual_rag_chain(memory)
    
    # Process with context
    answer = rag_chain.invoke({
        "question": question,
        "messages": messages
    })
    
    # Get sources with context
    sources = get_pdf_sources_scores_contextual(question, messages, memory)
    
    return answer, sources

# ─── 13. Fixed Custom CSS Styling ────────────────────────────────────────────
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
    
    /* Custom title styling */
    .main-title {
        text-align: center;
        padding: 2rem 0 1rem 0;
        font-size: 2.5rem;
        font-weight: 600;
        color: #34495E;
        border-bottom: 2px solid #E8E4DD;
        margin-bottom: 2rem;
    }
    
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
        position: relative;
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

# ─── 14. Fixed message display functions ─────────────────────────────────────
def display_user_message(content):
    st.markdown(f"""
    <div class="user-message">
        <div class="user-bubble">{content}</div>
    </div>
    """, unsafe_allow_html=True)

def display_assistant_message(content, sources=None, is_contextual=False):
    if is_contextual:
        st.markdown("""
        <div style="display: flex; justify-content: flex-start; margin-bottom: 5px;">
            <div style="background: linear-gradient(135deg, #FFF3E0, #FFE0B2); color: #E65100; padding: 4px 8px; border-radius: 10px; font-size: 0.7rem; font-weight: 500; box-shadow: 0 1px 3px rgba(0,0,0,0.2);">
                🔗 Using conversation context
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="assistant-message">
        <div class="assistant-bubble">
            {content}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if sources:
        with st.expander("📚 Sources used (PDF & Score)", expanded=False):
            sources_html = ""
            for i, (pdf_name, score) in enumerate(sources, 1):
                sources_html += f"<p><strong>{i}. {pdf_name}</strong> — Score: {score:.4f}</p>"
            st.markdown(f'<div class="sources-container">{sources_html}</div>', unsafe_allow_html=True)

# ─── 15. Main Streamlit interface ────────────────────────────────────────────
st.set_page_config(
    page_title="Student Performance RAG Chat", 
    page_icon="🎓",
    layout="wide"
)

load_css()

# Initialize conversation memory
memory = get_conversation_memory()

st.markdown('<h1 class="main-title">Learning Analytics Predictive Assistant</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #7F8C8D; font-size: 1.1rem; margin-bottom: 2rem;">Ask questions about student performance research with conversation context</p>', unsafe_allow_html=True)

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
        
        # Display chat history with consistent styling
        for message in st.session_state.messages:
            if message["role"] == "user":
                display_user_message(message["content"])
            else:
                sources = message.get("sources", None)
                is_contextual = message.get("is_contextual", False)
                display_assistant_message(message["content"], sources, is_contextual)
        
        st.markdown('</div>', unsafe_allow_html=True)

# Chat input
if prompt := st.chat_input("Type your question here..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    display_user_message(prompt)
    
    # Check if question is contextual
    is_contextual = memory.is_contextual_question(prompt)
    
    # Process the question with context
    with st.spinner("Thinking..."):
        answer, sources = process_question_contextual(prompt, st.session_state.messages, memory)
    
    # Check if the response is valid
    if "fucking" not in str(answer).lower() and "not available" not in str(answer).lower():
        # Valid response - display it
        display_assistant_message(answer, sources, is_contextual)
        
        # Add to chat history
        st.session_state.messages.append({
            "role": "assistant", 
            "content": answer,
            "sources": sources,
            "is_contextual": is_contextual
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

# Enhanced debug sidebar
with st.sidebar:
    st.markdown("### Debug Info")
    st.write(f"Attempt count: {st.session_state.attempt_count}")
    st.write(f"Total messages: {len(st.session_state.messages)}")
    
    # Show if last question was contextual
    if st.session_state.messages:
        last_user_msg = None
        for msg in reversed(st.session_state.messages):
            if msg["role"] == "user":
                last_user_msg = msg["content"]
                break
        
        if last_user_msg:
            is_contextual = memory.is_contextual_question(last_user_msg)
            st.write(f"Last question contextual: {'✅' if is_contextual else '❌'}")
    
    st.markdown("### Conversational Memory")
    st.write("✅ Active with contextual understanding")
    st.write("🔗 Detects references to previous exchanges")
    st.write("🔄 Reformulates contextual questions")
    
    st.markdown("### Few-Shot Prompting")
    st.write("✅ Implemented with 6 examples")
    st.write("Examples help the model better understand the expected format.")
    
    # Show conversation context if available
    if len(st.session_state.messages) >= 2:
        with st.expander("📝 Conversation Context", expanded=False):
            context = memory.get_recent_context(st.session_state.messages, max_exchanges=2)
            if context:
                st.text_area("Current context:", context, height=200, disabled=True)