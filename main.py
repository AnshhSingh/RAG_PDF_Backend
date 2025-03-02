import torch
import shutil
import os
import uuid
import nest_asyncio
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from llama_cloud_services import LlamaParse
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Document
from llama_index.core.node_parser import SentenceSplitter  # Added for chunking
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from transformers import AutoModelForCausalLM, AutoTokenizer
from pydantic import BaseModel

# Apply Nest AsyncIO
nest_asyncio.apply()

# Load environment variables
load_dotenv()

# Optimize CUDA settings
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Model configuration
model_name = "HuggingFaceTB/SmolLM2-1.7B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Configure text splitting
node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)  # Better chunking

# Embedding model setup
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.text_splitter = node_parser  # Apply chunking globally

app = FastAPI()

UPLOAD_DIR = "uploads"
PARSED_DIR = "parsed_texts"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PARSED_DIR, exist_ok=True)

parser = LlamaParse(result_type="markdown")

# Initialize index with chunking
documents = []
parsed_files = [os.path.join(PARSED_DIR, f) for f in os.listdir(PARSED_DIR) if f.endswith(".txt")]

for file_path in parsed_files:
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    documents.extend(node_parser.split_text(text))  # Split existing texts into chunks

index = VectorStoreIndex.from_documents([Document(text=t) for t in documents])

def parse_pdf(file_path: str):
    """Parse PDF with enhanced error handling"""
    try:
        file_extractor = {".pdf": parser}
        documents = SimpleDirectoryReader(
            input_files=[file_path],
            file_extractor=file_extractor
        ).load_data()
        return "\n".join([doc.text for doc in documents]) if documents else ""
    except Exception as e:
        print(f"PDF parsing error: {str(e)}")
        return ""

@app.post("/upload/")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(400, "Only PDFs allowed")

    file_id = str(uuid.uuid4())
    file_path = os.path.join(UPLOAD_DIR, f"{file_id}.pdf")

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        if parsed_text := parse_pdf(file_path):
            # Split and index chunks
            text_chunks = node_parser.split_text(parsed_text)
            for chunk in text_chunks:
                index.insert(Document(text=chunk))
            
            # Save parsed content
            with open(os.path.join(PARSED_DIR, f"{file_id}.txt"), "w") as f:
                f.write(parsed_text)
            
            return {"file_id": file_id, "message": "PDF processed successfully"}
        
        raise HTTPException(500, "PDF parsing failed")
    except Exception as e:
        raise HTTPException(500, f"Processing error: {str(e)}")

class QueryRequest(BaseModel):
    query: str
    max_length: int = 200  # User-configurable response length

@app.post("/query/")
async def query(request: QueryRequest):
    if not request.query:
        raise HTTPException(400, "Query required")

    try:
        # Retrieve relevant context chunks
        retriever = index.as_retriever(similarity_top_k=2)  # Get top 2 chunks
        results = retriever.retrieve(request.query)
        context = "\n".join([n.node.text for n in results[:2]])

        # Create structured prompt
        prompt_template = f"""Summarize the context below to provide a concise answer to the query.
        Context: {context}
        Query: {request.query}
        Concise Answer:"""
        
        # Tokenize with truncation
        inputs = tokenizer(
            prompt_template,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(model.device)

        # Generate with optimized parameters
        outputs = model.generate(
            **inputs,
            max_new_tokens=request.max_length,
            temperature=0.3,
            top_p=0.9,
            repetition_penalty=1.1,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id
        )

        # Decode and clean response
        response = tokenizer.decode(
            outputs[0][inputs.input_ids.shape[-1]:],
            skip_special_tokens=True
        ).strip()

        return {"response": response, "context_used": context}
    
    except Exception as e:
        raise HTTPException(500, f"Query error: {str(e)}")