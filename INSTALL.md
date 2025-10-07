# Installation Guide for RAG Q&A System

## System Requirements

- **Python**: 3.8 or higher
- **Memory**: Minimum 4GB RAM (8GB+ recommended)
- **Storage**: At least 2GB free space for models and data
- **Operating System**: Windows, macOS, or Linux

## Step-by-Step Installation

### 1. Python Environment Setup

#### Option A: Using Conda (Recommended)
```bash
# Create a new conda environment
conda create -n rag-qa python=3.9
conda activate rag-qa
```

#### Option B: Using venv
```bash
# Create virtual environment
python -m venv rag-qa-env

# Activate environment
# Windows:
rag-qa-env\Scripts\activate
# macOS/Linux:
source rag-qa-env/bin/activate
```

### 2. Install Dependencies

```bash
# Navigate to project directory
cd rag-qa-system

# Install all required packages
pip install -r requirements.txt
```

#### Manual Installation (if requirements.txt fails)
```bash
# Core dependencies
pip install streamlit==1.29.0
pip install pinecone-client==3.0.0
pip install transformers==4.36.2
pip install sentence-transformers==2.2.2
pip install langchain==0.1.0

# Document processing
pip install PyPDF2==3.0.1
pip install python-docx==1.1.0
pip install openpyxl==3.1.2
pip install pandas==2.1.4

# Database support
pip install sqlalchemy==2.0.25
pip install psycopg2-binary==2.9.9
pip install mysql-connector-python==8.2.0

# Visualization
pip install plotly==5.17.0
pip install matplotlib==3.8.2
pip install seaborn==0.13.0

# Additional packages
pip install streamlit-option-menu==0.3.6
pip install python-dotenv==1.0.0
pip install rank-bm25==0.2.2
```

### 3. Configuration Setup

#### Create Environment File
```bash
# Copy the example environment file
copy .env.example .env  # Windows
cp .env.example .env    # macOS/Linux
```

#### Edit the .env file
Open `.env` in a text editor and configure:

```env
# Required - Get from https://pinecone.io
PINECONE_API_KEY=your_actual_pinecone_api_key
PINECONE_ENVIRONMENT=us-east-1-aws

# Optional - for enhanced features
OPENAI_API_KEY=your_openai_api_key
HUGGING_FACE_API_TOKEN=your_hugging_face_token

# Database (optional)
DATABASE_URL=sqlite:///./rag_qa.db
```

### 4. Pinecone Account Setup

1. **Create Account**: Go to [pinecone.io](https://pinecone.io) and sign up
2. **Get API Key**: Navigate to API Keys in your dashboard
3. **Note Environment**: Check your environment (e.g., us-east-1-aws)
4. **Add to .env**: Update your `.env` file with these values

### 5. Verify Installation

```bash
# Test the installation
python -c "import streamlit, pinecone, transformers, pandas; print('All dependencies installed successfully!')"
```

### 6. Run the Application

```bash
# Start the Streamlit app
streamlit run app.py
```

The application should open in your browser at `http://localhost:8501`

## Database Setup (Optional)

### SQLite (Default)
No additional setup required - SQLite database files can be uploaded directly.

### MySQL Setup
```bash
# Install MySQL server (if not already installed)
# Windows: Download from mysql.com
# macOS: brew install mysql
# Ubuntu: sudo apt-get install mysql-server

# Create database
mysql -u root -p
CREATE DATABASE rag_qa_db;
```

### PostgreSQL Setup
```bash
# Install PostgreSQL (if not already installed)
# Windows: Download from postgresql.org
# macOS: brew install postgresql
# Ubuntu: sudo apt-get install postgresql

# Create database
sudo -u postgres createdb rag_qa_db
```

## Troubleshooting Installation

### Common Issues and Solutions

#### 1. **Package Installation Failures**
```bash
# Upgrade pip first
pip install --upgrade pip

# Install with no cache
pip install --no-cache-dir -r requirements.txt

# For specific package issues
pip install --force-reinstall package_name
```

#### 2. **PyTorch Installation Issues**
```bash
# Install PyTorch separately with CPU support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Or with CUDA support (if you have a compatible GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### 3. **Database Driver Issues**

For **PostgreSQL** on Windows:
```bash
# If psycopg2 fails, try:
pip install psycopg2-binary --no-binary psycopg2-binary
```

For **MySQL** connection issues:
```bash
# Alternative MySQL connector
pip install PyMySQL
```

#### 4. **Memory Issues During Model Loading**
```bash
# Set environment variable to reduce memory usage
export TRANSFORMERS_CACHE=/path/to/cache/directory
export HF_HOME=/path/to/huggingface/cache
```

#### 5. **Streamlit Port Issues**
```bash
# Run on different port
streamlit run app.py --server.port 8502

# Or specify in config
mkdir ~/.streamlit
echo "[server]" > ~/.streamlit/config.toml
echo "port = 8502" >> ~/.streamlit/config.toml
```

## Development Installation

For development with additional tools:

```bash
# Install development dependencies
pip install -r requirements.txt
pip install black flake8 pytest jupyter

# Install in editable mode
pip install -e .
```

## Docker Installation (Alternative)

Create a `Dockerfile`:

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address", "0.0.0.0"]
```

Build and run:
```bash
docker build -t rag-qa-system .
docker run -p 8501:8501 -v $(pwd)/.env:/app/.env rag-qa-system
```

## Performance Optimization

### For Better Performance:
1. **Use SSD storage** for faster file I/O
2. **Allocate more RAM** if processing large documents
3. **Use GPU** if available for model inference
4. **Configure Pinecone** with appropriate index size

### Resource Usage:
- **Minimum**: 4GB RAM, 2GB storage
- **Recommended**: 8GB+ RAM, 5GB+ storage
- **With GPU**: Additional 2-4GB VRAM

## Verification Checklist

- [ ] Python 3.8+ installed
- [ ] All dependencies installed successfully
- [ ] `.env` file configured with Pinecone credentials
- [ ] Streamlit application starts without errors
- [ ] Can access application at localhost:8501
- [ ] Settings page accepts Pinecone configuration
- [ ] Document upload works correctly
- [ ] (Optional) Database connection successful

## Getting Help

If you encounter issues:

1. **Check System Requirements**: Ensure your system meets minimum requirements
2. **Update Dependencies**: Run `pip install --upgrade -r requirements.txt`
3. **Clear Cache**: Remove `__pycache__` folders and restart
4. **Check Logs**: Look at terminal output for specific error messages
5. **Environment**: Verify your virtual environment is activated

For persistent issues, check the troubleshooting section in the main README.md file.