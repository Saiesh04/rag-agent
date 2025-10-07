# 🔧 Fixing Installation Issues

## Problem
The error indicates that `setuptools.build_meta` cannot be imported, which is a common issue with Python virtual environments that have missing or corrupted build tools.

## Solution: Step-by-Step Manual Installation

### Step 1: Fix Build Tools
```powershell
python -m pip install --upgrade pip
python -m pip install --upgrade setuptools wheel
```

### Step 2: Install Core Dependencies First
```powershell
pip install streamlit pandas numpy
```

### Step 3: Install PyTorch (CPU version for compatibility)
```powershell
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Step 4: Install Document Processing Libraries
```powershell
pip install PyPDF2 python-docx openpyxl
```

### Step 5: Install ML and NLP Libraries
```powershell
pip install transformers sentence-transformers
```

### Step 6: Install Vector Database Support
```powershell
pip install pinecone-client
```

### Step 7: Install Database Support
```powershell
pip install sqlalchemy
pip install psycopg2-binary
pip install mysql-connector-python
```

### Step 8: Install Visualization Libraries
```powershell
pip install plotly matplotlib seaborn
```

### Step 9: Install Remaining Utilities
```powershell
pip install python-dotenv requests scikit-learn
pip install nltk spacy rank-bm25
pip install streamlit-option-menu Pillow
```

## Alternative: Use the Automated Script

Run the provided script:
```powershell
.\install-fix.bat
```

## Alternative: Fresh Virtual Environment

If issues persist, create a fresh virtual environment:

```powershell
# Deactivate current environment
deactivate

# Remove old environment
rmdir /s .venv

# Create new environment
python -m venv .venv

# Activate new environment
.venv\Scripts\activate

# Install build tools first
python -m pip install --upgrade pip setuptools wheel

# Then install packages step by step as above
```

## Minimal Installation for Testing

If you want to test the basic functionality first:

```powershell
pip install streamlit pandas plotly python-dotenv
```

Then run:
```powershell
streamlit run app.py
```

You can add more packages as needed once the basic system is working.

## Common Issues and Solutions

### Issue: PyTorch Installation Fails
**Solution**: Use CPU-only version:
```powershell
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Issue: psycopg2 Installation Fails on Windows
**Solution**: Use binary version:
```powershell
pip install psycopg2-binary
```

### Issue: Some packages still fail
**Solution**: Install one by one and skip problematic ones temporarily:
```powershell
# Install essential packages first
pip install streamlit pandas plotly
pip install PyPDF2 python-docx openpyxl
pip install python-dotenv

# Test the app
streamlit run app.py

# Add more packages gradually
```

The system is designed to work even with missing optional packages - it will show warnings but won't crash.