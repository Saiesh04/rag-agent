## 🔑 How to Get a New Pinecone API Key

### Step 1: Access Pinecone Console
1. Go to [https://app.pinecone.io/](https://app.pinecone.io/)
2. Sign in to your account

### Step 2: Navigate to API Keys
1. Once logged in, look for **"API Keys"** in the left sidebar
2. Click on it to access your API key management page

### Step 3: Generate New API Key
1. Click **"Create API Key"** or **"New API Key"**
2. Give it a name (e.g., "RAG-QA-System")
3. Copy the generated key immediately (it won't be shown again)

### Step 4: Update Your Environment
1. Open the `.env` file in your project
2. Replace the current API key with the new one:
   ```
   PINECONE_API_KEY=your_new_api_key_here
   ```
3. Save the file

### Step 5: Test the New Key
1. Run the test script: `python test_pinecone.py`
2. If successful, try the Streamlit app again

### 🚨 Important Notes:
- **API keys expire** - check if your account is still active
- **Free tier limits** - ensure you haven't exceeded quotas
- **Copy carefully** - make sure no extra characters are included
- **No quotes needed** - don't wrap the API key in quotes in the .env file

### 🔍 Common Issues:
- **Account suspended** - check your Pinecone account status
- **Billing issues** - ensure payment method is valid
- **Region restrictions** - verify your account's region settings
- **Project permissions** - make sure the API key has proper permissions

### 💡 Alternative Solution:
If you continue having issues, try creating a **completely new Pinecone account** with a different email address to ensure a fresh start.