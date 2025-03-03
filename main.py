
import torch
import shutil
import os
import uuid
import nest_asyncio
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, HTTPException
from dotenv import load_dotenv
from llama_cloud_services import LlamaParse
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Document, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings, PromptTemplate
from transformers import AutoTokenizer
from pydantic import BaseModel
import traceback
from llama_index.llms.huggingface import HuggingFaceLLM
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware
from pyngrok import ngrok
import uvicorn
from threading import Thread
from llama_index.core import load_index_from_storage
from supabase import create_client

# Apply async fixes and load environment variables
nest_asyncio.apply()
load_dotenv()

# Validate required environment variables
required_env_vars = [
    "LLAMA_CLOUD_API_KEY",
    "SUPABASE_URL",
    "SUPABASE_KEY"
]
for var in required_env_vars:
    if not os.getenv(var):
        raise ValueError(f"{var} environment variable not set")

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# CUDA optimizations
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Text processing: Initialize sentence splitter
node_parser = SentenceSplitter(
    chunk_size=512,
    chunk_overlap=50,
    paragraph_separator="\n\n",
    secondary_chunking_regex="[^.,;。，]+[,.;。，]?",
)

# Set up embedding model
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5",
    embed_batch_size=16,
)

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    padding_side="left",
    use_fast=True
)

# LLM configuration and prompt template
query_wrapper_prompt = PromptTemplate(
    "Below is an instruction that describes a task.\n"
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{query_str}\n\n### Response:"
)

# Determine torch dtype based on CUDA availability
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

llm = HuggingFaceLLM(
    context_window=2048,
    max_new_tokens=512,
    generate_kwargs={
        "temperature": 0.25,
        "do_sample": False,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id
    },
    tokenizer=tokenizer,
    model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    device_map="auto",
    model_kwargs={
        "torch_dtype": torch_dtype,
    }
)
Settings.llm = llm

# Initialize FastAPI app with CORS
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set up directories
PDF_DIR = "pdf"
PARSED_DIR = "parsed_texts"
INDEX_DIR = "index_storage"
os.makedirs(PDF_DIR, exist_ok=True)
os.makedirs(PARSED_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)

parser = LlamaParse(result_type="markdown")

# Initialize or load vector index
if os.path.exists(os.path.join(INDEX_DIR, "docstore.json")):
    storage_context = StorageContext.from_defaults(persist_dir=INDEX_DIR)
    index = load_index_from_storage(storage_context)
else:
    documents = []
    for filename in os.listdir(PARSED_DIR):
        if filename.endswith(".txt"):
            with open(os.path.join(PARSED_DIR, filename), "r", encoding="utf-8") as f:
                text = f.read()
            documents.append(Document(text=text))
    
    if documents:
        nodes = node_parser.get_nodes_from_documents(documents)
        storage_context = StorageContext.from_defaults()
        storage_context.docstore.add_documents(nodes)
        index = VectorStoreIndex(nodes, storage_context=storage_context)
        storage_context.persist(persist_dir=INDEX_DIR)
    else:
        index = VectorStoreIndex([])
        index.storage_context.persist(persist_dir=INDEX_DIR)

def parse_pdf(file_path: str) -> str:
    try:
        file_extractor = {".pdf": parser}
        docs = SimpleDirectoryReader(
            input_files=[file_path],
            file_extractor=file_extractor,
            filename_as_id=True
        ).load_data()
        return "\n".join(doc.text for doc in docs) if docs else ""
    except Exception as e:
        raise RuntimeError(f"PDF parsing failed: {str(e)}")

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are allowed")

    file_id = uuid.uuid4().hex
    file_path = os.path.join(PDF_DIR, f"{file_id}.pdf")
    supabase_success = False

    try:
        # Save PDF file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Initialize Supabase client
        supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

        # Prepare metadata
        upload_data = {
            "file_id": file_id,
            "filename": file.filename,
            "upload_date": datetime.utcnow().isoformat(),
            "file_path": file_path
        }

        # Insert metadata into Supabase
        supabase_response = supabase.table("documents").insert(upload_data).execute()

        # Debugging: Print the entire response
        print("Supabase response:", supabase_response)

        # Check for errors in the response
        if not supabase_response.data:
            raise RuntimeError(f"Supabase error: {supabase_response}")

        supabase_success = True

        # Process document
        parsed_text = parse_pdf(file_path)
        if not parsed_text:
            raise HTTPException(500, "PDF parsing returned empty content")

        parsed_path = os.path.join(PARSED_DIR, f"{file_id}.txt")
        with open(parsed_path, "w", encoding="utf-8") as f:
            f.write(parsed_text)

        document = Document(text=parsed_text)
        nodes = node_parser.get_nodes_from_documents([document])
        index.insert_nodes(nodes)
        index.storage_context.persist(persist_dir=INDEX_DIR)

        return {"file_id": file_id, "message": "PDF processed successfully"}

    except Exception as e:
        # Cleanup PDF if metadata storage failed
        if not supabase_success and os.path.exists(file_path):
            os.remove(file_path)
        traceback.print_exc()
        raise HTTPException(500, f"Processing error: {str(e)}")

class QueryRequest(BaseModel):
    query: str
    max_length: Optional[int] = 500

@app.post("/query")
async def query(request: QueryRequest):
    if not request.query.strip():
        raise HTTPException(400, "Query cannot be empty")

    try:
        query_engine = index.as_query_engine(similarity_top_k=5)
        response = query_engine.query(request.query)
        response_text = response.response if response else "No relevant information found."

        tokens = tokenizer.encode(response_text, return_tensors="pt")[0]
        truncated_tokens = tokens[:min(len(tokens), request.max_length)]
        final_response = tokenizer.decode(truncated_tokens, skip_special_tokens=True)

        return {"response": final_response}

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"Query failed: {str(e)}")

def run_server():
    uvicorn.run(app, host="0.0.0.0", port=8001)

server_thread = Thread(target=run_server, daemon=True)
server_thread.start()

public_url = ngrok.connect(8001).public_url
print("Public URL:", public_url)
