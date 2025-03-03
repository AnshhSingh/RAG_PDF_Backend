# Setup
## This is the easiest way to setup and run the backend without a lot of time Folllow the steps below:
- Visit [https://colab.research.google.com/drive/1Xha2XgoOQQ8oLurqG8-g0LyrIONF7fkd?usp=sharing](https://colab.research.google.com/drive/1xCC7svPvqoGIx3nEy0kV9Y-fz0nDxHYx#scrollTo=WxxEIslimCDn)
- Run each cells 
- add the ngrok token and api key as explained in the notebook
- your backend is now ready and should give you a ngrok public URL
- Read [This](https://github.com/AnshhSingh/RAG_PDF_frontend) to connect the front-end

# API Documentation

## Overview
This API allows users to upload and parse PDFs, then query indexed document content using LlamaIndex to process request 

**Reference:** [LlamaParse Example huggingface](https://github.com/run-llama/llama_cloud_services/blob/main/parse.md)

### Base URL
```
http://xyzabc.ngrok.free
```

## Endpoints

### 1. Upload PDF

#### `POST /upload`
Parses an uploaded PDF file and indexes its content for querying 

#### Request
- **Headers:**
  - `Content-Type: multipart/form-data`
- **Body:**
  - `file` (Required) - A PDF file to be uploaded.

#### Response
```json
{
  "file_id": "<unique_file_id>",
  "message": "PDF processed successfully"
}
```
#### Errors
- `400` - Invalid file type (only PDFs are allowed).
- `500` - Parsing or storage failure.

**Reference:** [LlamaParse Documentation](https://github.com/run-llama/llama_cloud_services/blob/main/parse.md)

---

### 2. Query Indexed Documents

#### `POST /query`
Executes a natural language query against indexed document content.

#### Request
- **Headers:**
  - `Content-Type: application/json`
- **Body:**
  ```json
  {
    "query": "<your_query>",
    "max_length": 500
  }
  ```
  - `query` (Required) - The question to ask.
  - `max_length` (Optional) - Maximum length of the response (default: 500 characters).

#### Response
```json
{
  "response": "<generated_response>"
}
```
#### Errors
- `400` - Empty query.
- `500` - Query processing failure.

**Reference:** [LlamaIndex Query Engine](https://docs.llamaindex.ai/en/stable/examples/customization/llms/SimpleIndexDemo-Huggingface_camel/)

## Environment Variables
Ensure the following environment variables are set:
- `LLAMA_CLOUD_API_KEY`
- `SUPABASE_URL`
- `SUPABASE_KEY`

## Dependencies
- FastAPI
- LlamaIndex
- HuggingFace Transformers
- Supabase
- PyTorch
- Ngrok

## Running the API
Start the server using:
```sh
uvicorn main:app --host 0.0.0.0 --port 8001
```

