"""
Simple demo version of RAG Q&A System that works with minimal dependencies
Run this first to test basic functionality
"""
import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
import os

# Page configuration
st.set_page_config(
    page_title="RAG Q&A System - Demo",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        padding: 1rem;
        background: linear-gradient(90deg, #f0f2f6, #ffffff);
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .feature-card {
        background: white;
        color: #333333;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    
    .feature-card h3 {
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    
    .feature-card ul, .feature-card ol {
        color: #444444;
    }
    
    .feature-card li {
        margin-bottom: 0.5rem;
        color: #555555;
    }
    
    /* Ensure all text is visible */
    .stMarkdown {
        color: #333333;
    }
    
    /* Fix sidebar text */
    .sidebar .sidebar-content {
        color: #333333;
    }
    
    /* Ensure form text is visible */
    .stTextInput > div > div > input {
        color: #333333;
    }
    
    /* Fix selectbox text */
    .stSelectbox > div > div > div {
        color: #333333;
    }
    
    /* Fix button text */
    .stButton > button {
        color: #ffffff;
        background-color: #1f77b4;
        border: none;
    }
    
    /* Fix expander text */
    .streamlit-expanderHeader {
        color: #333333 !important;
    }
    
    /* Fix metric text */
    .metric-container {
        color: #333333;
    }
    
    /* Force dark text for all content */
    .main .block-container {
        color: #333333;
    }
    
    /* Fix any remaining text visibility issues */
    p, div, span, h1, h2, h3, h4, h5, h6 {
        color: #333333 !important;
    }
</style>
""", unsafe_allow_html=True)

def main():
    st.markdown('<div class="main-header">🤖 RAG Q&A System - Demo Version</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🚀 Demo Features")
        page = st.selectbox("Choose a demo:", [
            "🏠 Welcome",
            "📄 File Upload Demo", 
            "📊 Data Visualization Demo",
            "💬 Chat Interface Demo",
            "⚙️ System Status"
        ])
        
        st.markdown("---")
        st.markdown("### 📋 Installation Status")
        
        # Check for packages
        packages = {
            "Streamlit": True,
            "Pandas": True,
            "Plotly": True,
            "PyPDF2": False,
            "Transformers": False,
            "Pinecone": False
        }
        
        try:
            import PyPDF2
            packages["PyPDF2"] = True
        except ImportError:
            pass
            
        try:
            import transformers
            packages["Transformers"] = True
        except ImportError:
            pass
            
        try:
            import pinecone
            packages["Pinecone"] = True
        except ImportError:
            pass
        
        for pkg, installed in packages.items():
            status = "✅" if installed else "❌"
            st.write(f"{status} {pkg}")
    
    # Main content based on selection
    if page == "🏠 Welcome":
        welcome_page()
    elif page == "📄 File Upload Demo":
        file_upload_demo()
    elif page == "📊 Data Visualization Demo":
        visualization_demo()
    elif page == "💬 Chat Interface Demo":
        chat_demo()
    elif page == "⚙️ System Status":
        system_status()

def welcome_page():
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h3>🎯 What This Demo Shows</h3>
            <ul>
                <li><strong>Basic UI Structure</strong> - Modern Streamlit interface</li>
                <li><strong>File Upload Interface</strong> - Document handling simulation</li>
                <li><strong>Data Visualization</strong> - Interactive charts with Plotly</li>
                <li><strong>Chat Interface</strong> - Q&A conversation flow</li>
                <li><strong>System Status</strong> - Package installation checking</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h3>🔧 Next Steps</h3>
            <ol>
                <li><strong>Test this demo</strong> - Explore all the demo pages</li>
                <li><strong>Install packages</strong> - Run reset-and-install.bat</li>
                <li><strong>Configure API keys</strong> - Add Pinecone credentials</li>
                <li><strong>Run full app</strong> - Use the complete system</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

def file_upload_demo():
    st.markdown("### 📄 File Upload Demo")
    
    uploaded_files = st.file_uploader(
        "Upload demo files",
        accept_multiple_files=True,
        type=['pdf', 'docx', 'txt', 'csv']
    )
    
    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} file(s) uploaded successfully!")
        
        for file in uploaded_files:
            with st.expander(f"📄 {file.name}"):
                st.write(f"**File size:** {file.size:,} bytes")
                st.write(f"**File type:** {file.type}")
                
                if file.type == "text/csv":
                    df = pd.read_csv(file)
                    st.write("**Preview:**")
                    st.dataframe(df.head())
                    
                elif file.type == "text/plain":
                    content = str(file.read(), "utf-8")
                    st.write("**Content preview:**")
                    st.text(content[:500] + "..." if len(content) > 500 else content)

def visualization_demo():
    st.markdown("### 📊 Data Visualization Demo")
    
    # Generate sample data
    import numpy as np
    
    chart_type = st.selectbox("Choose chart type:", [
        "Line Chart", "Bar Chart", "Scatter Plot", "Histogram"
    ])
    
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    data = pd.DataFrame({
        'date': dates,
        'value1': np.random.randn(100).cumsum() + 100,
        'value2': np.random.randn(100).cumsum() + 50,
        'category': np.random.choice(['A', 'B', 'C'], 100)
    })
    
    if chart_type == "Line Chart":
        fig = px.line(data, x='date', y=['value1', 'value2'], 
                     title="Time Series Data")
        st.plotly_chart(fig, use_container_width=True)
        
    elif chart_type == "Bar Chart":
        category_data = data.groupby('category')['value1'].mean().reset_index()
        fig = px.bar(category_data, x='category', y='value1',
                    title="Average Values by Category")
        st.plotly_chart(fig, use_container_width=True)
        
    elif chart_type == "Scatter Plot":
        fig = px.scatter(data, x='value1', y='value2', color='category',
                        title="Value1 vs Value2")
        st.plotly_chart(fig, use_container_width=True)
        
    elif chart_type == "Histogram":
        fig = px.histogram(data, x='value1', nbins=20,
                          title="Distribution of Value1")
        st.plotly_chart(fig, use_container_width=True)
    
    st.write("**Sample Data:**")
    st.dataframe(data.head(10))

def chat_demo():
    st.markdown("### 💬 Chat Interface Demo")
    
    # Initialize chat history
    if 'demo_messages' not in st.session_state:
        st.session_state.demo_messages = []
    
    # Chat input
    user_input = st.text_input("Ask a question:", placeholder="What would you like to know?")
    
    if st.button("Send") and user_input:
        # Add user message
        st.session_state.demo_messages.append({
            "role": "user",
            "content": user_input,
            "timestamp": datetime.now()
        })
        
        # Generate demo response
        demo_responses = [
            "This is a demo response. In the full version, this would be generated by the RAG pipeline using your documents.",
            "The system would search through your uploaded documents to find relevant information and provide a detailed answer.",
            "With the complete setup, you'll get responses with source citations and confidence scores.",
            "This interface shows how the conversational aspect of the system works."
        ]
        
        import random
        response = random.choice(demo_responses)
        
        st.session_state.demo_messages.append({
            "role": "assistant", 
            "content": response,
            "timestamp": datetime.now()
        })
    
    # Display chat history
    for message in st.session_state.demo_messages:
        if message["role"] == "user":
            st.markdown(f"**You:** {message['content']}")
        else:
            st.markdown(f"**Assistant:** {message['content']}")
            st.markdown("---")
    
    if st.button("Clear Chat"):
        st.session_state.demo_messages = []
        st.rerun()

def system_status():
    st.markdown("### ⚙️ System Status")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📦 Python Packages")
        
        packages = [
            ("streamlit", "Streamlit"),
            ("pandas", "Pandas"), 
            ("plotly", "Plotly"),
            ("numpy", "NumPy"),
            ("PyPDF2", "PyPDF2"),
            ("docx", "python-docx"),
            ("openpyxl", "OpenPyXL"),
            ("transformers", "Transformers"),
            ("sentence_transformers", "Sentence Transformers"),
            ("pinecone", "Pinecone")
        ]
        
        for module_name, display_name in packages:
            try:
                __import__(module_name)
                st.success(f"✅ {display_name}")
            except ImportError:
                st.error(f"❌ {display_name}")
    
    with col2:
        st.markdown("#### 🔧 System Information")
        
        import sys
        import platform
        
        st.write(f"**Python Version:** {sys.version}")
        st.write(f"**Platform:** {platform.system()} {platform.release()}")
        st.write(f"**Working Directory:** {os.getcwd()}")
        
        # Environment variables
        st.markdown("#### 🔑 Environment Variables")
        env_vars = ["PINECONE_API_KEY", "OPENAI_API_KEY", "HUGGING_FACE_API_TOKEN"]
        
        for var in env_vars:
            value = os.getenv(var)
            if value:
                st.success(f"✅ {var}: {'*' * (len(value) - 4) + value[-4:]}")
            else:
                st.warning(f"⚠️ {var}: Not set")

if __name__ == "__main__":
    main()