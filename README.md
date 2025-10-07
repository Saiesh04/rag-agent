# RAG-Powered Question & Answer System

A comprehensive, LLM-powered Question & Answer system using Retrieval-Augmented Generation (RAG) with document processing, database connectivity, and advanced data visualization capabilities.

## 🚀 Features

### 📄 Document Processing
- **Multi-format Support**: PDF, DOCX, Excel (XLSX/XLS), and TXT files
- **Intelligent Chunking**: Automatic text segmentation with overlap
- **Metadata Extraction**: File-specific metadata preservation
- **Source Attribution**: Track answers back to original documents

### 🔍 Smart Search & Retrieval
- **Vector Search**: Powered by Pinecone vector database
- **Semantic Similarity**: Advanced embedding-based matching  
- **Re-ranking**: BM25 + vector similarity hybrid scoring
- **Context-Aware**: Maintains conversation context

### 🤖 LLM Integration
- **Flexible Models**: Support for various LLM models (currently using DialoGPT with Llama support)
- **RAG Pipeline**: Context-enhanced answer generation
- **Confidence Scoring**: Reliability assessment for responses
- **Source Highlighting**: Visual attribution in responses

### 🗄️ Database Connectivity
- **Multi-Database Support**: SQLite, MySQL, PostgreSQL
- **Interactive Connection**: Easy database setup via UI
- **Query Execution**: Natural language to SQL (basic implementation)
- **Data Analysis**: Automated insights and visualizations

### 📊 Advanced Analytics
- **Automated Analysis**: Statistical insights and patterns
- **Interactive Visualizations**: Plotly-powered charts and graphs
- **Data Profiling**: Column analysis, missing data detection
- **Correlation Analysis**: Relationship identification

### 🎨 Modern UI
- **Streamlit-Powered**: Responsive, modern interface
- **Multi-Page Navigation**: Organized workflow
- **Real-time Updates**: Live status indicators
- **Export Functionality**: Conversation history and analysis export

## 🛠️ Technology Stack

- **Frontend**: Streamlit with custom CSS
- **Vector Database**: Pinecone
- **LLM Framework**: Transformers, LangChain components
- **Document Processing**: PyPDF2, python-docx, openpyxl
- **Database**: SQLAlchemy with multiple database support
- **Visualization**: Plotly, Matplotlib, Seaborn
- **Data Processing**: Pandas, NumPy
- **Embeddings**: Sentence Transformers

## 📋 Prerequisites

- Python 3.8 or higher
- Pinecone account and API key
- (Optional) Database server for database connectivity features

## 🚀 Quick Start

### 1. Clone and Setup

```bash
# Navigate to the project directory
cd rag-qa-system

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

```bash
# Copy environment template
copy .env.example .env

# Edit .env file with your API keys
# Required: PINECONE_API_KEY
# Optional: OPENAI_API_KEY, HUGGING_FACE_API_TOKEN
```

### 3. Run the Application

```bash
# Start the Streamlit application
streamlit run app.py

# Or use Python directly
python app.py
```

The application will open in your browser at `http://localhost:8501`

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
# Required
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_ENVIRONMENT=us-east-1-aws

# Optional
OPENAI_API_KEY=your_openai_api_key_here
HUGGING_FACE_API_TOKEN=your_hugging_face_token_here
DATABASE_URL=sqlite:///./rag_qa.db
```

### Pinecone Setup

1. Create a Pinecone account at [pinecone.io](https://pinecone.io)
2. Get your API key from the console
3. Note your environment region
4. Configure in the application Settings page

## 📖 Usage Guide

### 1. Initial Setup
- Navigate to **Settings** page
- Enter your Pinecone API key and environment
- Save configuration to initialize the vector store

### 2. Document Management
- Go to **Document Manager** page
- Upload your documents (PDF, DOCX, Excel, TXT)
- Wait for processing and vector storage
- View document summaries and metadata

### 3. Database Connection (Optional)
- Visit **Database Connection** page
- Choose your database type (SQLite/MySQL/PostgreSQL)
- Enter connection details
- Explore tables and generate analyses

### 4. Ask Questions
- Navigate to **Ask Questions** page
- Enter your question in natural language
- Choose to search documents and/or database
- View answers with source attribution and confidence scores

### 5. Data Analysis
- Go to **Data Analysis** page
- View automated insights and visualizations
- Explore correlation matrices and distribution plots
- Export analysis results

## 🎯 Key Features in Detail

### Document Processing Pipeline
- **Chunking Strategy**: Intelligent text segmentation with sentence boundary detection
- **Metadata Preservation**: File-specific information (pages, sheets, etc.)
- **Deduplication**: Hash-based duplicate detection
- **Format Support**: Extensible document type support

### Vector Search System
- **Embedding Model**: Sentence Transformers for semantic similarity
- **Index Management**: Automated Pinecone index creation and management
- **Hybrid Scoring**: Combines vector similarity with BM25 ranking
- **Filtering**: Metadata-based result filtering

### Question Answering
- **Context Assembly**: Multi-document context aggregation
- **Answer Generation**: LLM-powered response generation
- **Source Attribution**: Detailed source tracking and highlighting
- **Confidence Assessment**: Reliability scoring for responses

### Database Integration
- **Connection Management**: Secure, persistent database connections
- **Schema Discovery**: Automatic table and column detection
- **Query Interface**: Basic natural language to SQL conversion
- **Analysis Pipeline**: Automated statistical analysis and visualization

## 🔧 Customization

### Adding New Document Types
Extend the `DocumentProcessor` class in `src/document_processor/processor.py`:

```python
def extract_text_from_newformat(self, file_path: str) -> Tuple[str, Dict[str, Any]]:
    # Implement extraction logic
    return text, metadata
```

### Custom LLM Models
Modify the `RAGPipeline` class in `src/llm/rag_pipeline.py`:

```python
def _initialize_model(self):
    # Add your model initialization
    self.qa_pipeline = your_model_pipeline
```

### Database Connectors
Extend `DatabaseConnector` in `src/database/connector.py` for new database types.

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**: Install all dependencies with `pip install -r requirements.txt`
2. **Pinecone Connection**: Verify API key and environment settings
3. **Memory Issues**: Reduce chunk size in configuration for large documents
4. **Model Loading**: Ensure sufficient disk space for model downloads

### Performance Optimization

- **Vector Store**: Use appropriate Pinecone index size for your data volume
- **Chunking**: Adjust chunk size/overlap based on document types
- **Database**: Use connection pooling for multiple database queries
- **Caching**: Implement result caching for frequently asked questions

## 📊 Monitoring and Analytics

The system provides built-in monitoring:
- Document processing statistics
- Query performance metrics
- Source reliability scoring
- Conversation history analysis

## 🔒 Security Considerations

- Store API keys securely in environment variables
- Use secure database connections (SSL/TLS)
- Implement rate limiting for production deployments
- Regular security updates for dependencies

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Pinecone for vector database services
- Hugging Face for transformer models
- Streamlit for the amazing web framework
- The open-source community for various libraries used

## 📞 Support

For issues and questions:
- Check the troubleshooting section
- Review the configuration guide
- Open an issue on the repository

---

**Built with ❤️ using Python, Streamlit, and modern AI technologies**