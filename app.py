import os
import uuid
import shutil
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from passlib.context import CryptContext
from sqlitedict import SqliteDict
import faiss
# import fitz
import pymupdf4llm

import ollama
from chromadb.utils.embedding_functions.ollama_embedding_function import OllamaEmbeddingFunction

###############################################################################
# Environment Setup - Ensure proper package versions and Ollama server running
###############################################################################

# Recommended installation commands:
# pip install --upgrade passlib==1.7.4 bcrypt==3.2.0
# pip install fastapi uvicorn python-multipart pydantic sqlitedict faiss-cpu pymupdf chromadb==0.3.26 langchain_community langchain_core langchain_text_splitters ollama

###############################################################################
# GLOBAL CONFIGURATION
###############################################################################

EMBEDDING_MODEL_NAME = "mxbai-embed-large:latest"
LLM_MODEL_NAME       = "deepseek-r1:1.5b"

app = FastAPI()

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

USER_DB_PATH  = "users.sqlite"
PLAN_DB_PATH  = "plans.sqlite"
CHUNK_DB_PATH = "chunks.sqlite"

VECTOR_INDEXES = {}

PDF_UPLOAD_FOLDER    = "pdf_uploads"
VECTOR_STORE_FOLDER  = "vector_store"

os.makedirs(PDF_UPLOAD_FOLDER, exist_ok=True)
os.makedirs(VECTOR_STORE_FOLDER, exist_ok=True)

SYSTEM_PROMPT = """You are an assistant that helps users understand their insurance documents.
Answer questions strictly based on the context you are provided with.
Give concise answers without formatting.
Use examples to help the user understand cost calculations better."""

###############################################################################
# DATA MODELS
###############################################################################

class User(BaseModel):
    username: str
    password: str

class LoginRequest(BaseModel):
    username: str
    password: str

class Plan(BaseModel):
    plan_id: str
    plan_name: str
    plan_type: str

class ChatRequest(BaseModel):
    user_id: str
    plan_id: str
    question: str

###############################################################################
# OLLAMA EMBEDDINGS & GENERATION
###############################################################################

ollama_embedding_function = OllamaEmbeddingFunction(
    url="http://localhost:11434/api/embeddings",
    model_name=EMBEDDING_MODEL_NAME,
)

def embed_text_with_ollama(text: str) -> np.ndarray:
    try:
        embedding_list = ollama_embedding_function(f"Represent this sentence for searching relevant passages: {text}")
        emb_array = np.array(embedding_list, dtype=np.float32)
        if emb_array.ndim == 2 and emb_array.shape[0] == 1:
            emb_array = emb_array[0]
        return emb_array
    except Exception as e:
        print(f"Error in embedding: {e}")
        return np.array([], dtype=np.float32)

def generate_text_with_ollama(prompt: str) -> str:
    response_stream = ollama.chat(
        model=LLM_MODEL_NAME,
        stream=True,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    final_text = ""
    for chunk in response_stream:
        if "message" in chunk and "content" in chunk["message"]:
            final_text += chunk["message"]["content"]
        if chunk.get("done"):
            break
    return final_text.strip()

###############################################################################
# FAISS UTILITIES
###############################################################################

def create_or_load_faiss_index(user_id: str, plan_id: str, embedding_dim: int) -> faiss.IndexFlatL2:
    key = (user_id, plan_id)
    if key not in VECTOR_INDEXES:
        index = faiss.IndexFlatL2(embedding_dim)
        VECTOR_INDEXES[key] = (index, embedding_dim, [])
    return VECTOR_INDEXES[key][0]

def add_vector_to_index(user_id: str, plan_id: str, embedding: np.ndarray, metadata: dict):
    key = (user_id, plan_id)
    if key not in VECTOR_INDEXES:
        index = faiss.IndexFlatL2(len(embedding))
        VECTOR_INDEXES[key] = (index, len(embedding), [])
    index, dim, metadatas = VECTOR_INDEXES[key]
    if len(embedding) != dim:
        print(f"[ERROR] Dimension mismatch: expected {dim}, got {len(embedding)}")
        return
    vec = embedding[np.newaxis, :]
    index.add(vec)
    metadatas.append(metadata)

def search_vectors(user_id: str, plan_id: str, query_embedding: np.ndarray, k: int = 5):
    key = (user_id, plan_id)
    if key not in VECTOR_INDEXES:
        return []
    index, dim, metadatas = VECTOR_INDEXES[key]
    if len(query_embedding) != dim:
        print(f"[ERROR] Query dim mismatch: expected {dim}, got {len(query_embedding)}")
        return []
    query_vec = query_embedding[np.newaxis, :]
    D, I = index.search(query_vec, k)
    results = []
    for dist, idx in zip(D[0], I[0]):
        if idx < 0 or idx >= len(metadatas):
            continue
        results.append((metadatas[idx], dist))
    return results

###############################################################################
# FASTAPI ENDPOINTS
###############################################################################

@app.post("/register")
def register_user(username: str = Form(...), password: str = Form(...)):
    with SqliteDict(USER_DB_PATH) as user_db:
        if username in user_db:
            return {"success": False, "message": "Username exists"}
        hashed = pwd_context.hash(password)
        user_id = str(uuid.uuid4())
        user_db[username] = {"user_id": user_id, "hashed_password": hashed}
        user_db.commit()
    return {"success": True, "user_id": user_id}

@app.post("/login")
def login_user(username: str = Form(...), password: str = Form(...)):
    with SqliteDict(USER_DB_PATH) as user_db:
        if username not in user_db:
            return {"success": False, "message": "User not found"}
        data = user_db[username]
        if pwd_context.verify(password, data["hashed_password"]):
            return {"success": True, "user_id": data["user_id"]}
        else:
            return {"success": False, "message": "Incorrect password"}

@app.post("/add_plan")
def add_plan(user_id: str = Form(...), plan_name: str = Form(...), plan_type: str = Form(...)):
    plan_id = str(uuid.uuid4())
    plan_data = {"plan_id": plan_id, "plan_name": plan_name, "plan_type": plan_type, "pdfs": []}
    with SqliteDict(PLAN_DB_PATH) as plan_db:
        plans = plan_db.get(user_id, [])
        plans.append(plan_data)
        plan_db[user_id] = plans
        plan_db.commit()
    return {"success": True, "plan_id": plan_id}

@app.post("/delete_plan")
def delete_plan(user_id: str = Form(...), plan_id: str = Form(...)):
    with SqliteDict(PLAN_DB_PATH) as plan_db:
        plans = plan_db.get(user_id, [])
        plans = [p for p in plans if p["plan_id"] != plan_id]
        plan_db[user_id] = plans
        plan_db.commit()

    key = (user_id, plan_id)
    if key in VECTOR_INDEXES:
        del VECTOR_INDEXES[key]

    with SqliteDict(CHUNK_DB_PATH) as chunk_db:
        ck = f"{user_id}:{plan_id}"
        if ck in chunk_db:
            del chunk_db[ck]
            chunk_db.commit()

    index_file = os.path.join(VECTOR_STORE_FOLDER, f"{user_id}_{plan_id}.faiss")
    if os.path.exists(index_file):
        os.remove(index_file)

    plan_folder = os.path.join(PDF_UPLOAD_FOLDER, user_id, plan_id)
    if os.path.isdir(plan_folder):
        shutil.rmtree(plan_folder, ignore_errors=True)

    return {"success": True}

@app.get("/get_plans")
def get_plans(user_id: str):
    with SqliteDict(PLAN_DB_PATH) as plan_db:
        return plan_db.get(user_id, [])

@app.post("/upload_pdf")
def upload_pdf(user_id: str = Form(...), plan_id: str = Form(...), file: UploadFile = File(...)):
    pdf_id = str(uuid.uuid4())
    plan_pdf_folder = os.path.join(PDF_UPLOAD_FOLDER, user_id, plan_id)
    os.makedirs(plan_pdf_folder, exist_ok=True)

    filename = f"{plan_id}_{pdf_id}.pdf"
    pdf_path = os.path.join(plan_pdf_folder, filename)

    pdf_bytes = file.file.read()
    with open(pdf_path, "wb") as f:
        f.write(pdf_bytes)

    with SqliteDict(PLAN_DB_PATH) as plan_db:
        plans = plan_db.get(user_id, [])
        for plan in plans:
            if plan["plan_id"] == plan_id:
                if "pdfs" not in plan:
                    plan["pdfs"] = []
                plan["pdfs"].append({"pdf_id": pdf_id, "filename": filename})
                break
        plan_db[user_id] = plans
        plan_db.commit()

    return {"success": True, "pdf_id": pdf_id}

@app.get("/get_pdfs")
def get_pdfs(user_id: str, plan_id: str):
    with SqliteDict(PLAN_DB_PATH) as plan_db:
        plans = plan_db.get(user_id, [])
        for plan in plans:
            if plan["plan_id"] == plan_id:
                return plan.get("pdfs", [])
    return []

@app.post("/embed_pdf")
def embed_pdf(user_id: str = Form(...), plan_id: str = Form(...), pdf_id: str = Form(...)):
    filename = None
    with SqliteDict(PLAN_DB_PATH) as plan_db:
        plans = plan_db.get(user_id, [])
        for plan in plans:
            if plan["plan_id"] == plan_id:
                for pdf_info in plan.get("pdfs", []):
                    if pdf_info["pdf_id"] == pdf_id:
                        filename = pdf_info["filename"]
                        break
    if not filename:
        return {"success": False, "message": "PDF not found."}

    plan_pdf_folder = os.path.join(PDF_UPLOAD_FOLDER, user_id, plan_id)
    pdf_path = os.path.join(plan_pdf_folder, filename)
    if not os.path.exists(pdf_path):
        return {"success": False, "message": "PDF file missing."}

    # doc = fitz.open(pdf_path)
    # docs = []
    # for page in doc:
    #     text = page.get_text()
    #     if text.strip():
    #         docs.append(text)

    # # Write the text to a file
    # with open(pdf_path.replace(".pdf", ".md"), "w", encoding='utf-8') as f:
    #     f.write("\n\n".join(docs))
    # print("Writing document to text file...")


    docs = pymupdf4llm.to_markdown(pdf_path, page_chunks=True)

    chunk_key = f"{user_id}:{plan_id}"

    with SqliteDict(CHUNK_DB_PATH) as chunk_db:
        for page_idx, page in enumerate(docs):
            # Extract page text (Markdown text from the new page dict)
            text = page.get("text", "")
            if not text:
                # Skip pages without text
                continue

            # Generate the embedding
            embedding = embed_text_with_ollama(text)
            print(f"Generated Embeddings for page {page_idx} : {text}")
            if embedding.shape[0] == 0:
                # Skip empty embeddings
                continue
            
            # Prepare metadata
            metadata = {
                "document_id": pdf_id,
                "page_number": page_idx,
                "page_metadata": page.get("metadata", {}),
                "toc_items": page.get("toc_items", []),
                "tables": page.get("tables", []),
                "images": page.get("images", []),
                "graphics": page.get("graphics", []),
            }
            
            # Add to Faiss index
            add_vector_to_index(user_id, plan_id, embedding, metadata)
            
            # Prepare and store chunk information in the SQLite DB
            chunk_id = str(uuid.uuid4())
            chunk_info = {
                "chunk_id": chunk_id,
                "page_number": page_idx,
                "chunk_text": text,
                "embedding": embedding.tolist(),
                "pdf_id": pdf_id,
                "metadata": metadata
            }
            
            # chunk_key can be (user_id, plan_id) or another key you prefer
            stored_chunks = chunk_db.get(chunk_key, [])
            stored_chunks.append(chunk_info)
            chunk_db[chunk_key] = stored_chunks
            chunk_db.commit()
            
            print(f"Stored Page {page_idx} with Chunk ID: {chunk_id}, Text length: {len(text)}")
        
        # After embedding, store the Faiss index
        key = (user_id, plan_id)
        if key in VECTOR_INDEXES:
            index, dim, _ = VECTOR_INDEXES[key]
            index_file = os.path.join(VECTOR_STORE_FOLDER, f"{user_id}_{plan_id}.faiss")
            faiss.write_index(index, index_file)
            print("FAISS index updated and stored.")

    return {"success": True, "message": "Embedding complete."}

@app.post("/load_embeddings")
def load_embeddings(user_id: str = Form(...), plan_id: str = Form(...)):
    key = (user_id, plan_id)
    index_file = os.path.join(VECTOR_STORE_FOLDER, f"{user_id}_{plan_id}.faiss")
    if not os.path.exists(index_file):
        return {"success": False, "message": "No stored embeddings found."}
    loaded_index = faiss.read_index(index_file)
    dim = loaded_index.d
    chunk_key = f"{user_id}:{plan_id}"
    with SqliteDict(CHUNK_DB_PATH) as chunk_db:
        stored_chunks = chunk_db.get(chunk_key, [])
    metadatas = []
    for c in stored_chunks:
        metadatas.append({
            "document_id": c.get("pdf_id"),
            "page_number": c.get("page_number"),
            "chunk_text": c.get("chunk_text"),
        })
    VECTOR_INDEXES[key] = (loaded_index, dim, metadatas)
    return {"success": True, "message": "Embeddings loaded from storage."}

@app.post("/chat")
def chat_with_plan(data: ChatRequest):
    key = (data.user_id, data.plan_id)
    if key not in VECTOR_INDEXES:
        index_file = os.path.join(VECTOR_STORE_FOLDER, f"{data.user_id}_{data.plan_id}.faiss")
        if os.path.exists(index_file):
            loaded_index = faiss.read_index(index_file)
            dim = loaded_index.d
            chunk_key = f"{data.user_id}:{data.plan_id}"
            with SqliteDict(CHUNK_DB_PATH) as chunk_db:
                stored_chunks = chunk_db.get(chunk_key, [])
            metadatas = []
            for c in stored_chunks:
                metadatas.append({
                    "document_id": c.get("pdf_id"),
                    "page_number": c.get("page_number"),
                    "chunk_text": c.get("chunk_text"),
                })
            print(f"Metadata loaded for {len(metadatas)} chunks.")   
            VECTOR_INDEXES[key] = (loaded_index, dim, metadatas)
        else:
            return {"answer": "No embeddings loaded or stored for this plan."}

    q_emb = embed_text_with_ollama(data.question)
    if q_emb.shape[0] == 0:
        return {"answer": "Error: Could not embed question."}

    top_chunks = search_vectors(data.user_id, data.plan_id, q_emb, k=5)
    context_texts = []
    for meta, _ in top_chunks:
        context_texts.append(f"(Page {meta['page_number']}) {meta['chunk_text']}")
    context_joined = "\n\n".join(context_texts)

    prompt = f"Context:\n{context_joined}\n\nQuestion:\n{data.question}"
    print(f"Prompt: {prompt}")
    
    answer = generate_text_with_ollama(prompt)
    return {"answer": answer}

@app.post("/purge_all_data")
def purge_all_data():
    VECTOR_INDEXES.clear()
    for path in [USER_DB_PATH, PLAN_DB_PATH, CHUNK_DB_PATH]:
        if os.path.exists(path):
            os.remove(path)
    if os.path.exists(PDF_UPLOAD_FOLDER):
        shutil.rmtree(PDF_UPLOAD_FOLDER, ignore_errors=True)
    os.makedirs(PDF_UPLOAD_FOLDER, exist_ok=True)
    if os.path.exists(VECTOR_STORE_FOLDER):
        shutil.rmtree(VECTOR_STORE_FOLDER, ignore_errors=True)
    os.makedirs(VECTOR_STORE_FOLDER, exist_ok=True)
    return {"success": True, "message": "All data purged."}

###############################################################################
# SIMPLE FRONTEND
###############################################################################

@app.get("/", response_class=HTMLResponse)
def serve_index():
    html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>Insurance Plan Document Search & Chat</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        fieldset { margin-bottom: 20px; }
        #login, #register, #plans, #pdfSection, #chatSection, #purgeSection {
            border: 1px solid #ccc; padding: 10px; margin-bottom: 20px;
        }
        .hidden { display: none; }
        #chatHistory { border: 1px solid #aaa; padding: 10px; height: 300px; overflow-y: scroll; margin-bottom: 10px; }
        .userMsg { color: blue; }
        .botMsg { color: green; }
        .loading { color: orange; }
    </style>
</head>
<body>
    <h1>Insurance Plan Document Search & Chat</h1>

    <div id="purgeSection">
        <h2>Purge All Data</h2>
        <button id="purgeBtn" style="background-color:red;color:white;">Purge All!</button>
        <div id="purgeResult"></div>
    </div>

    <div id="register">
        <h2>Register</h2>
        <form id="registerForm">
            <label>Username: <input name="username" type="text" /></label>
            <label>Password: <input name="password" type="password" /></label>
            <button type="submit">Register</button>
        </form>
        <div id="registerResult"></div>
    </div>

    <div id="login">
        <h2>Login</h2>
        <form id="loginForm">
            <label>Username: <input name="username" type="text" /></label>
            <label>Password: <input name="password" type="password" /></label>
            <button type="submit">Login</button>
        </form>
        <div id="loginResult"></div>
    </div>

    <div id="plans" class="hidden">
        <h2>Add Plan</h2>
        <form id="planForm">
            <label>Plan Name: <input name="plan_name" type="text" /></label><br/>
            <label>Plan Type: <input name="plan_type" type="text" /></label><br/>
            <button type="submit">Add Plan</button>
        </form>
        <div id="planResult"></div>
        <br/>
        <button id="loadPlansBtn">Load My Plans</button>
        <div id="plansList"></div>
    </div>

    <div id="pdfSection" class="hidden">
        <h2>PDF Management</h2>
        <form id="pdfUploadForm">
            <fieldset>
                <legend>Upload PDF</legend>
                <label>Select Plan:</label>
                <select id="planSelect" name="plan_id"></select><br/>
                <label>PDF file: <input id="pdfFile" name="file" type="file" /></label><br/>
                <button type="submit">Upload PDF</button>
            </fieldset>
        </form>
        <div id="uploadResult"></div>

        <fieldset>
            <legend>Plan PDFs</legend>
            <label>Select Plan:</label>
            <select id="pdfPlanSelect" name="pdfPlanSelect"></select>
            <button id="loadPdfsBtn">Load PDFs</button>
            <div id="pdfsList"></div>
        </fieldset>

        <fieldset>
            <legend>Embed a PDF</legend>
            <label>Select PDF to embed:</label>
            <select id="embedPdfSelect"></select>
            <button id="embedPdfBtn">Embed Selected PDF</button>
            <div id="embedResult"></div>
        </fieldset>

        <fieldset>
            <legend>Load Stored Embeddings</legend>
            <label>Select Plan:</label>
            <select id="loadPlanSelect"></select>
            <button id="loadEmbeddingsBtn">Load Embeddings</button>
            <div id="loadEmbeddingsResult"></div>
        </fieldset>
    </div>

    <div id="chatSection" class="hidden">
        <h2>Chat With Plan</h2>
        <div id="chatHistory"></div>
        <form id="chatForm">
            <label>Select Plan:</label>
            <select id="chatPlanSelect" name="plan_id"></select><br/>
            <textarea name="question" rows="2" cols="50" placeholder="Enter your question..."></textarea><br/>
            <button type="submit">Send</button>
        </form>
    </div>

    <script>
    let currentUserId = null;

    document.getElementById('purgeBtn').addEventListener('click', async () => {
        if (!confirm("Are you sure you want to delete ALL data?")) return;
        const resp = await fetch('/purge_all_data', { method: 'POST' });
        const data = await resp.json();
        document.getElementById('purgeResult').innerText = JSON.stringify(data);
    });

    document.getElementById('registerForm').addEventListener('submit', async (e) => {
        e.preventDefault();
        const formData = new FormData(e.target);
        const resp = await fetch('/register', { method: 'POST', body: formData });
        const data = await resp.json();
        document.getElementById('registerResult').innerText = JSON.stringify(data);
    });

    document.getElementById('loginForm').addEventListener('submit', async (e) => {
        e.preventDefault();
        const formData = new FormData(e.target);
        const resp = await fetch('/login', { method: 'POST', body: formData });
        const data = await resp.json();
        document.getElementById('loginResult').innerText = JSON.stringify(data);
        if (data.success) {
            currentUserId = data.user_id;
            document.getElementById('plans').classList.remove('hidden');
            document.getElementById('pdfSection').classList.remove('hidden');
            document.getElementById('chatSection').classList.remove('hidden');
        }
    });

    document.getElementById('planForm').addEventListener('submit', async (e) => {
        e.preventDefault();
        if (!currentUserId) { alert("Please login first"); return; }
        const formData = new FormData(e.target);
        formData.append("user_id", currentUserId);
        const resp = await fetch('/add_plan', { method: 'POST', body: formData });
        const data = await resp.json();
        document.getElementById('planResult').innerText = JSON.stringify(data);
    });

    document.getElementById('loadPlansBtn').addEventListener('click', async () => {
        if (!currentUserId) { alert("Please login first"); return; }
        const resp = await fetch('/get_plans?user_id=' + currentUserId);
        const data = await resp.json();
        const plansList = document.getElementById('plansList');
        plansList.innerHTML = '';

        const planSelect = document.getElementById('planSelect');
        const pdfPlanSelect = document.getElementById('pdfPlanSelect');
        const chatPlanSelect = document.getElementById('chatPlanSelect');
        const loadPlanSelect = document.getElementById('loadPlanSelect');

        planSelect.innerHTML = '';
        pdfPlanSelect.innerHTML = '';
        chatPlanSelect.innerHTML = '';
        loadPlanSelect.innerHTML = '';

        data.forEach(plan => {
            const div = document.createElement('div');
            div.innerHTML = `Plan ID: ${plan.plan_id}, Name: ${plan.plan_name}, Type: ${plan.plan_type}`;
            plansList.appendChild(div);

            let opt1 = document.createElement('option');
            opt1.value = plan.plan_id;
            opt1.text = plan.plan_name;
            planSelect.add(opt1);

            let opt2 = document.createElement('option');
            opt2.value = plan.plan_id;
            opt2.text = plan.plan_name;
            pdfPlanSelect.add(opt2);

            let opt3 = document.createElement('option');
            opt3.value = plan.plan_id;
            opt3.text = plan.plan_name;
            chatPlanSelect.add(opt3);

            let opt4 = document.createElement('option');
            opt4.value = plan.plan_id;
            opt4.text = plan.plan_name;
            loadPlanSelect.add(opt4);
        });
    });

    document.getElementById('pdfUploadForm').addEventListener('submit', async (e) => {
        e.preventDefault();
        if (!currentUserId) { alert("Please login first"); return; }
        const formData = new FormData(e.target);
        formData.append("user_id", currentUserId);
        const resp = await fetch('/upload_pdf', { method: 'POST', body: formData });
        const data = await resp.json();
        document.getElementById('uploadResult').innerText = JSON.stringify(data);
    });

    document.getElementById('loadPdfsBtn').addEventListener('click', async () => {
        const planId = document.getElementById('pdfPlanSelect').value;
        if (!planId || !currentUserId) { alert("Select a plan."); return; }
        const resp = await fetch(`/get_pdfs?user_id=${currentUserId}&plan_id=${planId}`);
        const data = await resp.json();
        const pdfsList = document.getElementById('pdfsList');
        pdfsList.innerHTML = '';
        const embedPdfSelect = document.getElementById('embedPdfSelect');
        embedPdfSelect.innerHTML = '';

        data.forEach(pdf => {
            const div = document.createElement('div');
            div.innerHTML = `PDF ID: ${pdf.pdf_id}, Filename: ${pdf.filename}`;
            pdfsList.appendChild(div);

            let opt = document.createElement('option');
            opt.value = pdf.pdf_id;
            opt.text = pdf.filename;
            embedPdfSelect.add(opt);
        });
    });

    document.getElementById('embedPdfBtn').addEventListener('click', async () => {
        const planId = document.getElementById('pdfPlanSelect').value;
        const pdfId  = document.getElementById('embedPdfSelect').value;
        if (!planId || !pdfId || !currentUserId) {
            alert("Select plan/pdf and login.");
            return;
        }
        document.getElementById('embedResult').innerText = "Embedding...";
        const formData = new FormData();
        formData.append("user_id", currentUserId);
        formData.append("plan_id", planId);
        formData.append("pdf_id", pdfId);
        const resp = await fetch('/embed_pdf', { method: 'POST', body: formData });
        const data = await resp.json();
        document.getElementById('embedResult').innerText = data.message;
    });

    document.getElementById('loadEmbeddingsBtn').addEventListener('click', async () => {
        const planId = document.getElementById('loadPlanSelect').value;
        if (!planId || !currentUserId) { alert("Select a plan."); return; }
        document.getElementById('loadEmbeddingsResult').innerText = "Loading embeddings...";
        const formData = new FormData();
        formData.append("user_id", currentUserId);
        formData.append("plan_id", planId);
        const resp = await fetch('/load_embeddings', { method: 'POST', body: formData });
        const data = await resp.json();
        document.getElementById('loadEmbeddingsResult').innerText = data.message;
    });

    const chatHistory = document.getElementById('chatHistory');
    document.getElementById('chatForm').addEventListener('submit', async (e) => {
        e.preventDefault();
        if (!currentUserId) { alert("Please login first"); return; }
        const formData = new FormData(e.target);
        const question = formData.get('question');
        const plan_id = formData.get('plan_id');
        if (!question.trim()) return;

        const userMsg = document.createElement('div');
        userMsg.className = 'userMsg';
        userMsg.innerText = "You: " + question;
        chatHistory.appendChild(userMsg);
        chatHistory.scrollTop = chatHistory.scrollHeight;

        const botMsg = document.createElement('div');
        botMsg.className = 'botMsg loading';
        botMsg.innerText = "Bot is typing...";
        chatHistory.appendChild(botMsg);
        chatHistory.scrollTop = chatHistory.scrollHeight;

        const reqData = { user_id: currentUserId, plan_id, question };
        const resp = await fetch('/chat', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(reqData)
        });
        const data = await resp.json();

        botMsg.classList.remove('loading');
        botMsg.innerText = "Bot: " + data.answer;
        chatHistory.scrollTop = chatHistory.scrollHeight;

        e.target.reset();
    });
    </script>
</body>
</html>
"""
    return HTMLResponse(content=html_content, status_code=200)
