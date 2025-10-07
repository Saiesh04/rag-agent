"""
Main Streamlit application for RAG Q&A System
"""
import streamlit as st
import os
import sys
import logging
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
import time
from datetime import datetime

# Add src to path for imports
current_dir = Path(__file__).parent
src_dir = current_dir.parent / "src"
sys.path.insert(0, str(src_dir))

# Import our modules with graceful fallbacks
DocumentProcessor = None
VectorStore = None
RAGPipeline = None
DatabaseConnector = None
DataVisualizer = None
config = None

missing_modules = []

try:
    from document_processor import DocumentProcessor
except ImportError as e:
    missing_modules.append("Document Processor")

try:
    from vector_store import VectorStore
except ImportError as e:
    missing_modules.append("Vector Store")
    
try:
    from llm import RAGPipeline
except ImportError as e:
    missing_modules.append("LLM Pipeline")
    
try:
    from database import DatabaseConnector
except ImportError as e:
    missing_modules.append("Database Connector")
    
try:
    from visualization import DataVisualizer
except ImportError as e:
    missing_modules.append("Data Visualizer")

try:
    from config.settings import config
except ImportError as e:
    # Create a basic config if settings can't be imported
    class BasicConfig:
        APP_TITLE = "RAG Q&A System"
        PAGE_ICON = "🤖"  
        LAYOUT = "wide"
    config = BasicConfig()

if missing_modules:
    st.warning(f"⚠️ Some modules are not available: {', '.join(missing_modules)}")
    st.info("The app will run with limited functionality. Install missing packages to enable all features.")

# Configure logging
logging.basicConfig(level=logging.INFO)

# Page configuration
st.set_page_config(
    page_title=config.APP_TITLE,
    page_icon=config.PAGE_ICON,
    layout=config.LAYOUT,
    initial_sidebar_state="expanded"
)

# Custom CSS for maximum UI visibility
st.markdown("""
<style>
    /* Import fonts for better readability */
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700;900&display=swap');
    
    /* FORCE ALL TEXT TO BE DARK AND VISIBLE */
    *, *::before, *::after {
        color: #000000 !important;
        font-family: 'Roboto', Arial, sans-serif !important;
        font-weight: 600 !important;
    }
    
    /* Main app background - pure white */
    .stApp {
        background-color: #ffffff !important;
        color: #000000 !important;
    }
    
    /* Main content area */
    .main .block-container {
        background-color: #ffffff !important;
        color: #000000 !important;
    }
    
    /* All text elements */
    h1, h2, h3, h4, h5, h6, p, span, div, li, a, label, 
    .stMarkdown, .stMarkdown *, .stText, .stText *,
    .stSelectbox, .stTextInput, .stTextArea, .stButton,
    .streamlit-expanderHeader, .streamlit-expanderContent {
        color: #000000 !important;
        font-weight: 700 !important;
        text-shadow: none !important;
        background-color: transparent !important;
    }
    
    /* Header styling */
    .main-header {
        font-size: 2.5rem !important;
        font-weight: 900 !important;
        color: #000000 !important;
        text-align: center;
        margin: 2rem 0;
        padding: 2rem;
        background: #f0f0f0 !important;
        border-radius: 10px;
        border: 3px solid #000000 !important;
    }
    
    /* Card styling */
    .feature-card {
        background: #f8f8f8 !important;
        color: #000000 !important;
        padding: 2rem;
        border-radius: 10px;
        border: 3px solid #000000 !important;
        margin: 1rem 0;
    }
    
    .feature-card * {
        color: #000000 !important;
        font-weight: 700 !important;
    }
    
    /* Message styling */
    .success-message {
        background: #e0ffe0 !important;
        color: #000000 !important;
        padding: 1rem;
        border-radius: 5px;
        border: 3px solid #008000 !important;
        margin: 1rem 0;
        font-weight: 900 !important;
        font-size: 16px !important;
    }
    
    .error-message {
        background: #ffe0e0 !important;
        color: #000000 !important;
        padding: 1rem;
        border-radius: 5px;
        border: 3px solid #ff0000 !important;
        margin: 1rem 0;
        font-weight: 900 !important;
        font-size: 16px !important;
    }
    
    .source-highlight {
        background: #fff0e0 !important;
        color: #000000 !important;
        padding: 1rem;
        border-radius: 5px;
        border: 3px solid #ff8800 !important;
        margin: 1rem 0;
        font-weight: 800 !important;
        font-size: 16px !important;
    }
    
    /* Metric card styling */
    .metric-card {
        background: #e0e0ff !important;
        color: #000000 !important;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem;
        border: 3px solid #000080 !important;
    }
    
    .metric-card h3 {
        font-size: 2rem !important;
        font-weight: 900 !important;
        margin-bottom: 0.5rem;
        color: #000000 !important;
    }
    
    .metric-card p {
        font-size: 1rem !important;
        font-weight: 700 !important;
        color: #000000 !important;
    }
    
    /* Sidebar styling - Multiple selectors to catch all variations */
    .css-1d391kg, 
    [data-testid="stSidebar"], 
    .css-1cypcdb,
    .css-17eq0hr,
    .css-1544g2n,
    section[data-testid="stSidebar"],
    .sidebar .sidebar-content {
        background-color: #f0f0f0 !important;
        border-right: 3px solid #000000 !important;
    }
    
    /* Force all sidebar text to be visible */
    .css-1d391kg *, 
    [data-testid="stSidebar"] *,
    .css-1cypcdb *,
    .css-17eq0hr *,
    .css-1544g2n *,
    section[data-testid="stSidebar"] *,
    .sidebar * {
        color: #000000 !important;
        font-weight: 900 !important;
        font-size: 16px !important;
        text-shadow: none !important;
        background-color: transparent !important;
    }
    
    /* Sidebar navigation menu specifically */
    .nav-link, 
    .nav-link-selected,
    .streamlit-option-menu *,
    [data-baseweb="tab"] *,
    [role="tab"] * {
        color: #000000 !important;
        font-weight: 900 !important;
        font-size: 16px !important;
        background-color: #ffffff !important;
        border: 2px solid #000000 !important;
        margin: 2px 0 !important;
        padding: 8px !important;
    }
    
    /* Selected navigation item */
    .nav-link-selected,
    [data-baseweb="tab"][aria-selected="true"] * {
        background-color: #e0e0ff !important;
        color: #000000 !important;
        font-weight: 900 !important;
    }
    
    /* Form labels */
    label, .stSelectbox label, .stTextInput label, .stTextArea label, .stFileUploader label {
        color: #000000 !important;
        font-weight: 900 !important;
        font-size: 16px !important;
    }
    
    /* Form inputs */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea,
    .stSelectbox > div > div > select {
        background-color: #ffffff !important;
        color: #000000 !important;
        border: 3px solid #000000 !important;
        border-radius: 5px;
        font-weight: 700 !important;
        font-size: 16px !important;
        padding: 10px !important;
    }
    
    .stTextInput > div > div > input::placeholder,
    .stTextArea > div > div > textarea::placeholder {
        color: #666666 !important;
        font-weight: 600 !important;
    }
    
    /* Button styling */
    .stButton > button {
        background: #0080ff !important;
        color: #ffffff !important;
        border: 3px solid #000000 !important;
        border-radius: 5px;
        padding: 12px 24px !important;
        font-weight: 900 !important;
        font-size: 16px !important;
    }
    
    .stButton > button:hover {
        background: #0060cc !important;
        color: #ffffff !important;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #f0f0f0 !important;
        color: #000000 !important;
        font-weight: 900 !important;
        border: 3px solid #000000 !important;
        border-radius: 5px;
    }
    
    .streamlit-expanderContent {
        background-color: #ffffff !important;
        color: #000000 !important;
        border: 3px solid #000000 !important;
    }
    
    /* Streamlit alerts and notifications */
    .stAlert, .stInfo, .stSuccess, .stWarning, .stError {
        color: #000000 !important;
        font-weight: 900 !important;
        background-color: #f0f0f0 !important;
        border: 3px solid #000000 !important;
    }
    
    /* Additional sidebar targeting */
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 {
        color: #000000 !important;
        font-weight: 900 !important;
        font-size: 18px !important;
        margin: 10px 0 !important;
    }
    
    /* Sidebar markdown content */
    .css-1d391kg .stMarkdown,
    [data-testid="stSidebar"] .stMarkdown,
    .css-1d391kg .stMarkdown *,
    [data-testid="stSidebar"] .stMarkdown * {
        color: #000000 !important;
        font-weight: 900 !important;
        font-size: 16px !important;
        background-color: #f8f8f8 !important;
        padding: 5px !important;
        border-left: 4px solid #000000 !important;
        margin: 5px 0 !important;
    }
    
    /* System status section */
    .css-1d391kg strong,
    [data-testid="stSidebar"] strong {
        color: #000000 !important;
        font-weight: 900 !important;
        font-size: 16px !important;
        background-color: #e0e0e0 !important;
        padding: 3px 6px !important;
        border-radius: 3px !important;
    }
    
    /* Streamlit option menu specific targeting */
    .css-nlntq9,
    .css-1v4eu3n,
    .css-1629p8f,
    [data-testid="stVerticalBlock"] {
        background-color: #f0f0f0 !important;
    }
    
    .css-nlntq9 *,
    .css-1v4eu3n *,
    .css-1629p8f *,
    [data-testid="stVerticalBlock"] * {
        color: #000000 !important;
        font-weight: 900 !important;
        font-size: 16px !important;
    }
    
    /* Option menu container */
    [data-testid="stVerticalBlock"] [data-testid="element-container"] {
        background-color: #ffffff !important;
        border: 2px solid #000000 !important;
        border-radius: 5px !important;
        margin: 5px 0 !important;
        padding: 10px !important;
    }
    
    /* Final override for all elements */
    [data-testid="stMarkdownContainer"] *, 
    [data-testid="stText"] *,
    [class*="st"] *,
    .element-container * {
        color: #000000 !important;
        font-weight: 700 !important;
    }
    
    /* Force light theme always */
    @media (prefers-color-scheme: dark) {
        .stApp {
            background-color: #ffffff !important;
        }
        
        * {
            color: #000000 !important;
            background-color: transparent !important;
        }
    }
    
    /* Additional specific overrides */
    .stProgress > div > div {
        background-color: #0080ff !important;
    }
    
    .stFileUploader label, .stCheckbox label, .stRadio label {
        color: #000000 !important;
        font-weight: 900 !important;
    }
    
    /* Status indicators */
    .stSpinner > div {
        border-color: #0080ff !important;
    }
    
    /* Make sure everything is super visible */
    body {
        background: #ffffff !important;
    }
    
    .main {
        background: #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def initialize_session_state():
    """Initialize session state variables"""
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = None
    if 'rag_pipeline' not in st.session_state:
        st.session_state.rag_pipeline = None
    if 'db_connector' not in st.session_state:
        st.session_state.db_connector = DatabaseConnector() if DatabaseConnector else None
    if 'data_visualizer' not in st.session_state:
        st.session_state.data_visualizer = DataVisualizer() if DataVisualizer else None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'processed_documents' not in st.session_state:
        st.session_state.processed_documents = []
    if 'db_connected' not in st.session_state:
        st.session_state.db_connected = False
    if 'pinecone_configured' not in st.session_state:
        st.session_state.pinecone_configured = False
    if 'last_error' not in st.session_state:
        st.session_state.last_error = None

def sidebar_navigation():
    """Create sidebar navigation"""
    with st.sidebar:
        st.markdown('''
            <div style="background: #e0e0e0; color: #000000; padding: 20px; border: 3px solid #000000; border-radius: 10px; text-align: center; margin-bottom: 20px;">
                <h1 style="color: #000000; font-weight: 900; font-size: 24px; margin: 0;">🤖 RAG Q&A System</h1>
            </div>
        ''', unsafe_allow_html=True)
        
        selected = option_menu(
            menu_title="Navigation",
            options=[
                "🏠 Home",
                "📄 Document Manager", 
                "🗄️ Database Connection",
                "❓ Ask Questions",
                "📊 Data Analysis",
                "⚙️ Settings"
            ],
            icons=['house', 'file-text', 'database', 'question-circle', 'bar-chart', 'gear'],
            menu_icon="cast",
            default_index=0,
            styles={
                "container": {"padding": "10px", "background-color": "#f0f0f0", "border": "3px solid #000000", "border-radius": "10px"},
                "icon": {"color": "#000000", "font-size": "20px", "font-weight": "900"},
                "nav-link": {
                    "font-size": "16px", 
                    "text-align": "left", 
                    "margin": "2px", 
                    "padding": "10px",
                    "color": "#000000",
                    "font-weight": "900",
                    "background-color": "#ffffff",
                    "border": "2px solid #000000",
                    "border-radius": "5px",
                    "--hover-color": "#e0e0e0"
                },
                "nav-link-selected": {
                    "background-color": "#0080ff", 
                    "color": "#ffffff",
                    "font-weight": "900",
                    "border": "3px solid #000000"
                },
            }
        )
        
        # System Status
        st.markdown('<hr style="border: 2px solid #000000; margin: 20px 0;">', unsafe_allow_html=True)
        st.markdown('''
            <div style="background: #f0f0f0; color: #000000; padding: 15px; border: 3px solid #000000; border-radius: 10px; margin: 10px 0;">
                <h3 style="color: #000000; font-weight: 900; font-size: 18px; margin-bottom: 15px;">📋 System Status</h3>
            </div>
        ''', unsafe_allow_html=True)
        
        # Vector Store Status
        vector_status = "✅ Connected" if st.session_state.vector_store else "❌ Not Connected"
        vector_color = "#008000" if st.session_state.vector_store else "#ff0000"
        st.markdown(f'''
            <div style="background: #ffffff; color: #000000; padding: 10px; border: 2px solid {vector_color}; border-radius: 5px; margin: 5px 0;">
                <strong style="color: #000000; font-weight: 900;">Vector Store:</strong> 
                <span style="color: {vector_color}; font-weight: 900;">{vector_status}</span>
            </div>
        ''', unsafe_allow_html=True)
        
        # Database Status
        db_status = "✅ Connected" if st.session_state.db_connected else "❌ Not Connected"
        db_color = "#008000" if st.session_state.db_connected else "#ff0000"
        st.markdown(f'''
            <div style="background: #ffffff; color: #000000; padding: 10px; border: 2px solid {db_color}; border-radius: 5px; margin: 5px 0;">
                <strong style="color: #000000; font-weight: 900;">Database:</strong> 
                <span style="color: {db_color}; font-weight: 900;">{db_status}</span>
            </div>
        ''', unsafe_allow_html=True)
        
        # Documents Status
        doc_count = len(st.session_state.processed_documents)
        st.markdown(f'''
            <div style="background: #ffffff; color: #000000; padding: 10px; border: 2px solid #0080ff; border-radius: 5px; margin: 5px 0;">
                <strong style="color: #000000; font-weight: 900;">Documents:</strong> 
                <span style="color: #0080ff; font-weight: 900;">{doc_count} processed</span>
            </div>
        ''', unsafe_allow_html=True)
        
    return selected

def home_page():
    """Display home page"""
    st.markdown('<div class="main-header">🤖 Welcome to RAG-Powered Q&A System</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h3>🚀 Key Features</h3>
            <ul>
                <li><strong>Multi-format Document Processing</strong> - PDF, DOCX, Excel, TXT</li>
                <li><strong>Advanced Vector Search</strong> - Powered by Pinecone</li>
                <li><strong>Smart Question Answering</strong> - Context-aware responses</li>
                <li><strong>Database Integration</strong> - Connect to SQL databases</li>
                <li><strong>Data Visualization</strong> - Automated charts and insights</li>
                <li><strong>Source Attribution</strong> - Track answer sources</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h3>🎯 How to Get Started</h3>
            <ol>
                <li><strong>Configure Settings</strong> - Set up your API keys</li>
                <li><strong>Upload Documents</strong> - Add your knowledge base</li>
                <li><strong>Connect Database</strong> - Optional data source</li>
                <li><strong>Ask Questions</strong> - Get intelligent answers</li>
                <li><strong>Analyze Data</strong> - Generate insights</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
    
    # Quick stats
    st.markdown("### 📊 Quick Stats")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>{}</h3>
            <p>Documents Processed</p>
        </div>
        """.format(len(st.session_state.processed_documents)), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>{}</h3>
            <p>Questions Asked</p>
        </div>
        """.format(len(st.session_state.chat_history)), unsafe_allow_html=True)
    
    with col3:
        vector_connected = 1 if st.session_state.vector_store else 0
        st.markdown("""
        <div class="metric-card">
            <h3>{}</h3>
            <p>Vector Store</p>
        </div>
        """.format("Connected" if vector_connected else "Disconnected"), unsafe_allow_html=True)
    
    with col4:
        db_connected = 1 if st.session_state.db_connected else 0
        st.markdown("""
        <div class="metric-card">
            <h3>{}</h3>
            <p>Database</p>
        </div>
        """.format("Connected" if db_connected else "Disconnected"), unsafe_allow_html=True)
    
    # Setup guidance
    if not st.session_state.vector_store:
        st.markdown("---")
        st.markdown("### 🚀 Quick Setup")
        st.markdown("""
        <div class="feature-card">
            <h4>🔧 Get Started in 3 Easy Steps:</h4>
            <ol>
                <li><strong>Get Pinecone API Key:</strong> Visit <a href="https://app.pinecone.io/" target="_blank">Pinecone Console</a> and create a free account</li>
                <li><strong>Configure Settings:</strong> Go to Settings page and enter your API key</li>
                <li><strong>Upload Documents:</strong> Add your PDF, DOCX, or Excel files in Document Manager</li>
            </ol>
            <p><em>💡 The free tier gives you plenty of storage to get started!</em></p>
        </div>
        """, unsafe_allow_html=True)

def document_manager_page():
    """Document management page"""
    st.markdown('<div class="main-header">📄 Document Manager</div>', unsafe_allow_html=True)
    
    # File upload section
    st.markdown("### 📤 Upload Documents")
    
    uploaded_files = st.file_uploader(
        "Choose files to upload",
        type=['pdf', 'docx', 'xlsx', 'xls', 'txt'],
        accept_multiple_files=True,
        help="Supported formats: PDF, DOCX, Excel, TXT"
    )
    
    if uploaded_files:
        st.markdown(f"**{len(uploaded_files)} file(s) selected for processing**")
        
        if st.button("🚀 Process Documents", type="primary"):
            process_documents(uploaded_files)
    
    # Document list
    st.markdown("### 📚 Processed Documents")
    
    if st.session_state.processed_documents:
        for i, doc in enumerate(st.session_state.processed_documents):
            with st.expander(f"📄 {doc['filename']} ({doc['file_type'].upper()})"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Size:** {doc['size']:,} characters")
                    st.write(f"**Chunks:** {doc['num_chunks']}")
                    st.write(f"**Processed:** {doc['processed_at'][:19]}")
                
                with col2:
                    if doc['file_type'] == 'pdf':
                        st.write(f"**Pages:** {doc['metadata'].get('pages', 'N/A')}")
                    elif doc['file_type'] == 'excel':
                        st.write(f"**Sheets:** {doc['metadata'].get('total_sheets', 'N/A')}")
                
                # Document summary
                if st.button(f"📋 Generate Summary", key=f"summary_{i}"):
                    generate_document_summary(doc)
                
                # Delete document
                if st.button(f"🗑️ Delete Document", key=f"delete_{i}", type="secondary"):
                    delete_document(doc, i)
    else:
        st.info("No documents processed yet. Upload some documents to get started!")

def process_documents(uploaded_files):
    """Process uploaded documents"""
    if not st.session_state.vector_store:
        st.error("Please configure vector store in Settings first!")
        return
    
    doc_processor = DocumentProcessor()
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, uploaded_file in enumerate(uploaded_files):
        try:
            status_text.text(f"Processing {uploaded_file.name}...")
            
            # Process document
            doc_data = doc_processor.process_uploaded_file(uploaded_file)
            
            # Store in vector database
            success = st.session_state.vector_store.upsert_documents(doc_data)
            
            if success:
                st.session_state.processed_documents.append(doc_data)
                st.success(f"✅ Successfully processed {uploaded_file.name}")
            else:
                st.error(f"❌ Failed to store {uploaded_file.name}")
                
        except Exception as e:
            st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")
        
        progress_bar.progress((i + 1) / len(uploaded_files))
    
    status_text.text("Processing complete!")
    time.sleep(1)
    status_text.empty()
    progress_bar.empty()

def generate_document_summary(doc_data):
    """Generate document summary"""
    if not st.session_state.rag_pipeline:
        st.error("RAG pipeline not initialized!")
        return
    
    with st.spinner("Generating summary..."):
        summary = st.session_state.rag_pipeline.summarize_document(doc_data)
        
        st.markdown("**📋 Document Summary:**")
        st.markdown(f'<div class="source-highlight">{summary}</div>', unsafe_allow_html=True)

def delete_document(doc_data, index):
    """Delete a document"""
    try:
        # Delete from vector store
        if st.session_state.vector_store:
            st.session_state.vector_store.delete_document(doc_data['hash'])
        
        # Remove from session state
        st.session_state.processed_documents.pop(index)
        
        st.success(f"✅ Deleted {doc_data['filename']}")
        st.rerun()
        
    except Exception as e:
        st.error(f"❌ Error deleting document: {str(e)}")

def database_connection_page():
    """Database connection page"""
    st.markdown('<div class="main-header">🗄️ Database Connection</div>', unsafe_allow_html=True)
    
    # Connection form
    st.markdown("### 🔌 Connect to Database")
    
    db_type = st.selectbox(
        "Database Type",
        ["SQLite", "MySQL", "PostgreSQL"],
        help="Select your database type"
    )
    
    if db_type == "SQLite":
        db_file = st.file_uploader(
            "Upload SQLite Database File",
            type=['db', 'sqlite', 'sqlite3'],
            help="Upload your SQLite database file"
        )
        
        if db_file:
            # Save uploaded file temporarily
            temp_path = f"temp_{db_file.name}"
            with open(temp_path, "wb") as f:
                f.write(db_file.getbuffer())
            
            if st.button("🔗 Connect to SQLite", type="primary"):
                connect_to_database("sqlite", db_path=temp_path)
    
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            host = st.text_input("Host", value="localhost")
            database = st.text_input("Database Name")
            
        with col2:
            port = st.number_input("Port", value=3306 if db_type == "MySQL" else 5432)
            username = st.text_input("Username")
        
        password = st.text_input("Password", type="password")
        
        if st.button(f"🔗 Connect to {db_type}", type="primary"):
            connect_to_database(db_type.lower(), host=host, port=port, 
                              database=database, username=username, password=password)
    
    # Connection status and database info
    if st.session_state.db_connected:
        st.markdown("### ✅ Database Connected")
        
        # Get database summary
        summary = st.session_state.db_connector.get_database_summary()
        
        if summary.get("success"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Database Type", summary["db_type"].upper())
                st.metric("Total Tables", summary["total_tables"])
            
            with col2:
                st.write("**Connection Info:**")
                for key, value in summary["connection_params"].items():
                    st.write(f"- {key.title()}: {value}")
            
            # Tables information
            st.markdown("### 📊 Tables Overview")
            
            for table_name, table_info in summary["tables"].items():
                with st.expander(f"📋 {table_name}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Rows", f"{table_info['rows']:,}")
                        st.metric("Columns", table_info['columns'])
                    
                    with col2:
                        st.write("**Column Details:**")
                        for col in table_info['column_details'][:10]:  # Show first 10 columns
                            st.write(f"- {col['name']} ({col['type']})")
                    
                    if st.button(f"📈 Analyze {table_name}", key=f"analyze_{table_name}"):
                        analyze_table(table_name)
        
        # Disconnect button
        if st.button("🔌 Disconnect Database", type="secondary"):
            disconnect_database()

def connect_to_database(db_type, **kwargs):
    """Connect to database"""
    try:
        with st.spinner("Connecting to database..."):
            if db_type == "sqlite":
                success = st.session_state.db_connector.connect_sqlite(kwargs['db_path'])
            elif db_type == "mysql":
                success = st.session_state.db_connector.connect_mysql(
                    kwargs['host'], kwargs['port'], kwargs['database'], 
                    kwargs['username'], kwargs['password']
                )
            elif db_type == "postgresql":
                success = st.session_state.db_connector.connect_postgresql(
                    kwargs['host'], kwargs['port'], kwargs['database'], 
                    kwargs['username'], kwargs['password']
                )
            
            if success:
                st.session_state.db_connected = True
                st.success("✅ Database connected successfully!")
                st.rerun()
            else:
                st.error("❌ Failed to connect to database")
                
    except Exception as e:
        st.error(f"❌ Connection error: {str(e)}")

def disconnect_database():
    """Disconnect from database"""
    st.session_state.db_connector.close_connection()
    st.session_state.db_connected = False
    st.success("✅ Database disconnected")
    st.rerun()

def analyze_table(table_name):
    """Analyze a specific table"""
    try:
        # Get table data
        query = f"SELECT * FROM {table_name} LIMIT 1000"  # Limit for performance
        result = st.session_state.db_connector.execute_query(query)
        
        if result["success"]:
            df = pd.DataFrame(result["data"])
            
            # Analyze data
            dashboard = st.session_state.data_visualizer.create_summary_dashboard(df, table_name)
            
            # Store in session state for data analysis page
            st.session_state.current_analysis = dashboard
            
            st.success(f"✅ Analysis complete for {table_name}. Check the Data Analysis page!")
        else:
            st.error(f"❌ Error querying table: {result['error']}")
            
    except Exception as e:
        st.error(f"❌ Analysis error: {str(e)}")

def ask_questions_page():
    """Question answering page"""
    st.markdown('<div class="main-header">❓ Ask Questions</div>', unsafe_allow_html=True)
    
    # Initialize RAG pipeline if not already done
    if not st.session_state.rag_pipeline:
        with st.spinner("Initializing RAG pipeline..."):
            try:
                st.session_state.rag_pipeline = RAGPipeline()
                st.success("✅ RAG pipeline initialized!")
            except Exception as e:
                st.error(f"❌ Error initializing RAG pipeline: {str(e)}")
                return
    
    # Check if vector store is available
    if not st.session_state.vector_store:
        st.warning("⚠️ Vector store not configured. Please set up in Settings first.")
        return
    
    if not st.session_state.processed_documents:
        st.warning("⚠️ No documents available. Please upload documents first.")
        return
    
    # Question input
    st.markdown("### 💬 Ask Your Question")
    
    question = st.text_area(
        "Enter your question:",
        height=100,
        placeholder="Ask anything about your documents or database..."
    )
    
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        search_docs = st.checkbox("📄 Search Documents", value=True)
    
    with col2:
        search_db = st.checkbox("🗄️ Query Database", value=st.session_state.db_connected)
    
    with col3:
        top_k = st.slider("Number of sources to retrieve", 1, 10, 5)
    
    if st.button("🚀 Get Answer", type="primary") and question:
        get_answer(question, search_docs, search_db, top_k)
    
    # Chat history
    st.markdown("### 💬 Chat History")
    
    for i, chat in enumerate(reversed(st.session_state.chat_history)):
        with st.expander(f"Q: {chat['question'][:100]}..." if len(chat['question']) > 100 else f"Q: {chat['question']}"):
            st.markdown(f"**Question:** {chat['question']}")
            st.markdown(f"**Answer:** {chat['answer']}")
            
            if chat.get('sources'):
                st.markdown("**Sources:**")
                for j, source in enumerate(chat['sources']):
                    st.markdown(f'<div class="source-highlight">📄 <strong>{source["filename"]}</strong> (Score: {source["score"]:.3f})<br>{source["text"][:200]}...</div>', unsafe_allow_html=True)
    
    if st.session_state.chat_history and st.button("🗑️ Clear History"):
        st.session_state.chat_history = []
        st.rerun()

def get_answer(question, search_docs, search_db, top_k):
    """Get answer for a question"""
    try:
        with st.spinner("Searching for relevant information..."):
            contexts = []
            
            # Search documents
            if search_docs and st.session_state.vector_store:
                doc_contexts = st.session_state.vector_store.search_similar(question, top_k)
                contexts.extend(doc_contexts)
            
            # Search database (simplified - in practice, you'd convert NL to SQL)
            if search_db and st.session_state.db_connected:
                # This is a placeholder - implement NL to SQL conversion
                st.info("Database querying is currently limited. Focusing on document search.")
            
            if not contexts:
                st.warning("No relevant information found.")
                return
            
            # Generate answer
            with st.spinner("Generating answer..."):
                result = st.session_state.rag_pipeline.generate_answer(question, contexts)
                
                # Display answer
                st.markdown("### 🎯 Answer")
                st.markdown(f'<div class="success-message">{result["answer"]}</div>', unsafe_allow_html=True)
                
                # Display confidence
                confidence = result.get("confidence", 0)
                st.progress(confidence)
                st.caption(f"Confidence: {confidence:.1%}")
                
                # Display sources
                if result.get("sources"):
                    st.markdown("### 📚 Sources")
                    
                    for i, source in enumerate(result["sources"]):
                        st.markdown(f"""
                        <div class="source-highlight">
                            <strong>📄 {source["filename"]}</strong> (Relevance: {source["score"]:.3f})<br>
                            <em>Section {source["chunk_id"] + 1}</em><br>
                            {source["text"][:300]}...
                        </div>
                        """, unsafe_allow_html=True)
                
                # Save to chat history
                st.session_state.chat_history.append({
                    "question": question,
                    "answer": result["answer"],
                    "sources": result.get("sources", []),
                    "confidence": confidence,
                    "timestamp": datetime.now().isoformat()
                })
                
    except Exception as e:
        st.error(f"❌ Error getting answer: {str(e)}")

def data_analysis_page():
    """Data analysis and visualization page"""
    st.markdown('<div class="main-header">📊 Data Analysis</div>', unsafe_allow_html=True)
    
    # Check if we have analysis data
    if 'current_analysis' not in st.session_state:
        st.info("No data analysis available. Connect to a database and analyze a table first.")
        return
    
    analysis = st.session_state.current_analysis
    
    if analysis.get("error"):
        st.error(f"Analysis error: {analysis['error']}")
        return
    
    st.markdown(f"### 📋 Analysis for: {analysis.get('table_name', 'Data')}")
    
    # Display insights
    if analysis.get("analysis", {}).get("insights"):
        st.markdown("### 🔍 Key Insights")
        insights_text = st.session_state.data_visualizer.generate_insights_text(analysis["analysis"])
        st.markdown(f'<div class="feature-card">{insights_text}</div>', unsafe_allow_html=True)
    
    # Display visualizations
    plots = analysis.get("plots", {})
    
    if plots:
        st.markdown("### 📈 Visualizations")
        
        # Create tabs for different plot types
        plot_types = list(plots.keys())
        tabs = st.tabs([plot_type.replace("_", " ").title() for plot_type in plot_types])
        
        for tab, plot_type in zip(tabs, plot_types):
            with tab:
                if plot_type in plots:
                    st.plotly_chart(plots[plot_type], use_container_width=True)
    
    # Basic statistics
    basic_info = analysis.get("analysis", {}).get("basic_info", {})
    if basic_info:
        st.markdown("### 📊 Basic Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Rows", f"{basic_info.get('shape', [0, 0])[0]:,}")
        
        with col2:
            st.metric("Columns", basic_info.get('shape', [0, 0])[1])
        
        with col3:
            memory_mb = basic_info.get('memory_usage', 0) / (1024 * 1024)
            st.metric("Memory Usage", f"{memory_mb:.1f} MB")
        
        with col4:
            null_count = sum(basic_info.get('null_counts', {}).values())
            st.metric("Missing Values", f"{null_count:,}")

def settings_page():
    """Settings and configuration page"""
    st.markdown('<div class="main-header">⚙️ Settings</div>', unsafe_allow_html=True)
    
    # API Configuration
    st.markdown("### 🔑 API Configuration")
    
    if not st.session_state.vector_store:
        st.markdown("""
        <div class="feature-card">
            <h4>🚀 Setting Up Your Pinecone Account</h4>
            <ol>
                <li><strong>Sign up:</strong> Visit <a href="https://app.pinecone.io/" target="_blank">Pinecone Console</a></li>
                <li><strong>Create Project:</strong> Once logged in, create a new project</li>
                <li><strong>Get API Key:</strong> Go to API Keys section and copy your key</li>
                <li><strong>Note Environment:</strong> Check your project's environment/region</li>
            </ol>
            <p><em>💡 Free tier includes 1 index with 1M vectors - perfect for getting started!</em></p>
        </div>
        """, unsafe_allow_html=True)
    
    with st.form("api_config"):
        pinecone_api_key = st.text_input(
            "Pinecone API Key",
            type="password",
            value=os.getenv("PINECONE_API_KEY", ""),
            help="Get your API key from Pinecone console - it should start with pc-, pcsk_, or pcsk-",
            placeholder="pcsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
        )
        
        pinecone_env = st.selectbox(
            "Pinecone Environment",
            options=[
                "us-east-1", 
                "us-west-2", 
                "eu-west-1", 
                "asia-southeast-1",
                "us-central1-gcp",
                "us-east1-gcp", 
                "us-west1-gcp",
                "europe-west1-gcp",
                "asia-northeast1-gcp"
            ],
            index=0,
            help="Select your Pinecone environment region from the dropdown"
        )
        
        index_name = st.text_input(
            "Pinecone Index Name",
            value="rag-qa-index",
            help="Name for your Pinecone index"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            submitted = st.form_submit_button("💾 Save Configuration")
        with col2:
            test_key = st.form_submit_button("🔍 Test API Key Only")
        
        if test_key:
            if pinecone_api_key and pinecone_api_key.strip():
                api_key_clean = pinecone_api_key.strip()
                
                # Basic validation
                valid_prefixes = ['pc-', 'pcsk_', 'pcsk-']
                if not any(api_key_clean.startswith(prefix) for prefix in valid_prefixes):
                    st.error("❌ Invalid API key format")
                    return
                
                # Test API key using the test method
                try:
                    with st.spinner("🔍 Testing API key..."):
                        if VectorStore:
                            result = VectorStore.test_api_key(api_key_clean)
                            if result["success"]:
                                st.success("✅ API key is valid and Pinecone connection successful!")
                                if result["indexes"]:
                                    st.info(f"📋 Found {len(result['indexes'])} existing indexes: {', '.join(result['indexes'])}")
                                else:
                                    st.info("📋 No existing indexes found - new index will be created.")
                            else:
                                st.error(f"❌ API key test failed: {result['error']}")
                        else:
                            st.error("❌ Vector store module not available")
                except Exception as e:
                    st.error(f"❌ API key test failed: {str(e)}")
                    error_str = str(e).lower()
                    if "401" in error_str or "unauthorized" in error_str:
                        st.info("💡 API key is invalid or expired. Please get a new one from Pinecone console.")
                    elif "network" in error_str or "timeout" in error_str:
                        st.info("💡 Network issue. Check your internet connection.")
            else:
                st.warning("⚠️ Please enter an API key to test")
        
        if submitted:
            if pinecone_api_key and pinecone_api_key.strip():
                # Basic API key validation
                api_key_clean = pinecone_api_key.strip()
                
                # Check for valid Pinecone API key formats
                valid_prefixes = ['pc-', 'pcsk_', 'pcsk-']
                if not any(api_key_clean.startswith(prefix) for prefix in valid_prefixes):
                    st.error("❌ Invalid API key format. Pinecone API keys should start with 'pc-', 'pcsk_', or 'pcsk-'")
                    st.info("💡 Make sure you're copying the API key (not the project ID) from Pinecone console.")
                    return
                
                if len(api_key_clean) < 30:
                    st.error("❌ API key seems too short. Please check you've copied the complete key.")
                    return
                
                # Initialize vector store
                try:
                    with st.spinner("🔗 Connecting to Pinecone... This may take a few minutes for new indexes."):
                        if not VectorStore:
                            st.error("❌ Vector store module not available. Please check installation.")
                            return
                            
                        st.session_state.vector_store = VectorStore(
                            api_key=api_key_clean,
                            environment=pinecone_env,
                            index_name=index_name.strip()
                        )
                    
                    st.success("✅ Vector store configured successfully!")
                    st.info("💡 Your vector store is now ready. You can upload documents in the Document Manager.")
                    
                    # Save configuration to session for persistence
                    st.session_state.pinecone_configured = True
                    
                    # Refresh the page to update status
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error configuring vector store: {str(e)}")
                    
                    # Provide helpful suggestions based on error type
                    error_str = str(e).lower()
                    if "api key" in error_str or "unauthorized" in error_str:
                        st.info("💡 **Troubleshooting:** Make sure your API key is correct and copied fully from Pinecone console.")
                        st.info("🔑 **API Key Format:** Should start with 'pc-', 'pcsk_', or 'pcsk-' followed by a long string")
                    elif "environment" in error_str or "not found" in error_str:
                        st.info("💡 **Troubleshooting:** Try a different environment region from the dropdown above.")
                        st.info("🌍 **Common Regions:** us-east-1, us-west-2, eu-west-1")
                    elif "quota" in error_str or "limit" in error_str:
                        st.info("💡 **Troubleshooting:** You may have reached your Pinecone free tier limits. Consider upgrading your plan.")
                    elif "network" in error_str or "timeout" in error_str:
                        st.info("💡 **Troubleshooting:** Check your internet connection and try again.")
                    
                    # Clear any partial initialization
                    st.session_state.vector_store = None
                    
                    # Show debug information
                    with st.expander("🔍 Debug Information"):
                        st.write("**API Key Info:**")
                        st.write(f"- Length: {len(api_key_clean)}")
                        st.write(f"- Starts with: {api_key_clean[:10]}...")
                        st.write(f"- Environment: {pinecone_env}")
                        st.write(f"- Index name: {index_name}")
                        st.write("**Error Details:**")
                        st.code(str(e))
                        
                        st.write("**Troubleshooting Steps:**")
                        st.write("1. **Test your API key first** using the 'Test API Key Only' button")
                        st.write("2. Double-check your API key from Pinecone console")  
                        st.write("3. Make sure you're copying the API key, not project ID")
                        st.write("4. Try creating a new API key")
                        st.write("5. Check if your Pinecone account is active")
                        st.write("6. Ensure you have Pinecone credits/quota available")
                        
                        st.info("💡 **Quick Test:** Use the 'Test API Key Only' button above to verify your API key works before trying to create the full vector store.")
                    
            else:
                st.warning("⚠️ Please provide a valid Pinecone API key")
                st.info("💡 Get your free API key from [Pinecone Console](https://app.pinecone.io/)")
    
    # Troubleshooting section
    st.markdown("---")
    st.markdown("### 🔧 Troubleshooting API Key Issues")
    
    if st.button("🧪 Run Detailed API Key Test"):
        st.info("🔄 Running comprehensive API key test...")
        
        # Run the test
        api_key = os.getenv("PINECONE_API_KEY", "")
        if api_key:
            st.write(f"**API Key Found:** ✅")
            st.write(f"**Length:** {len(api_key)} characters")
            st.write(f"**Format:** {'✅ Valid' if any(api_key.startswith(p) for p in ['pc-', 'pcsk_', 'pcsk-']) else '❌ Invalid'}")
            
            # Test connection
            try:
                if VectorStore:
                    result = VectorStore.test_api_key(api_key)
                    if result["success"]:
                        st.success("✅ API key is working correctly!")
                        st.write(f"Found {len(result['indexes'])} indexes: {result['indexes']}")
                    else:
                        st.error(f"❌ API key test failed: {result['error']}")
                        st.markdown("""
                        **Possible Solutions:**
                        1. 🔄 **Generate a new API key** from Pinecone console
                        2. 🧾 **Check your account status** - ensure it's active
                        3. 💳 **Verify billing** - free tier may have expired
                        4. 🌍 **Try different region** - some regions may be restricted
                        5. 📧 **Create new account** - if all else fails
                        """)
            except Exception as e:
                st.error(f"❌ Test failed: {str(e)}")
        else:
            st.error("❌ No API key found in environment")
    
    # Demo mode option
    st.markdown("---")
    st.markdown("### 🎭 Demo Mode")
    st.markdown("""
    <div class="feature-card">
        <p><strong>Want to try the app without setting up Pinecone?</strong></p>
        <p>Check out <code>demo.py</code> for a simplified version that works with local storage!</p>
        <p><em>Run: <code>streamlit run demo.py</code></em></p>
        <p><strong>Or create a new Pinecone account:</strong></p>
        <p>1. Visit <a href="https://app.pinecone.io/" target="_blank">Pinecone Console</a></p>
        <p>2. Sign up with a new email address</p>
        <p>3. Generate a fresh API key</p>
        <p>4. Update your .env file with the new key</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Model Configuration
    st.markdown("### 🤖 Model Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Current LLM Model:** microsoft/DialoGPT-medium")
        st.write("**Embedding Model:** all-MiniLM-L6-v2")
    
    with col2:
        st.write("**Chunk Size:** 1000 characters")
        st.write("**Chunk Overlap:** 200 characters")
    
    # Vector Store Status
    if st.session_state.vector_store:
        st.markdown("### 📊 Vector Store Status")
        
        try:
            stats = st.session_state.vector_store.get_index_stats()
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Vectors", f"{stats.get('total_vectors', 0):,}")
            
            with col2:
                st.metric("Dimension", stats.get('dimension', 0))
            
            with col3:
                fullness = stats.get('index_fullness', 0)
                st.metric("Index Fullness", f"{fullness:.1%}")
                
        except Exception as e:
            st.error(f"Error getting vector store stats: {str(e)}")

def main():
    """Main application function"""
    initialize_session_state()
    
    # Navigation
    selected_page = sidebar_navigation()
    
    # Route to appropriate page
    if selected_page == "🏠 Home":
        home_page()
    elif selected_page == "📄 Document Manager":
        document_manager_page()
    elif selected_page == "🗄️ Database Connection":
        database_connection_page()
    elif selected_page == "❓ Ask Questions":
        ask_questions_page()
    elif selected_page == "📊 Data Analysis":
        data_analysis_page()
    elif selected_page == "⚙️ Settings":
        settings_page()

if __name__ == "__main__":
    main()