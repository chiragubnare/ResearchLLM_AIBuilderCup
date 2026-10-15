# H2S AI Builder Cup — Research Paper LLM Chatbot

An intelligent LLM-powered research assistant that allows users to upload a research paper in PDF format and interact with it through a conversational chatbot.

The system processes the uploaded research paper, extracts text, tables, and images, converts the extracted information into a searchable knowledge base, and uses Retrieval-Augmented Generation (RAG) to provide context-grounded answers.

Answers must cite retrieved source markers. The app checks that every answer sentence cites a marker from the retrieved set and maps each marker to its actual PDF page. This validates citation references; it is not a separate semantic fact-checking model.

## Run the UI locally

The repository includes a Streamlit app for uploading a PDF, indexing its text/tables/figures, and asking questions with page citations.

1. Install Python 3.11 or newer.
2. Create and activate a virtual environment:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Install the Python packages:

   ```bash
   pip install -r requirements.txt
   ```

4. Copy `.env.example` to `.env` and add a Gemini API key from Google AI Studio.
5. Start the app:

   ```bash
   streamlit run app.py
   ```

Streamlit prints a local URL, usually `http://localhost:8501`. Upload a PDF in the sidebar, choose **Index paper**, then ask questions in the chat box.

By default, files are saved under the ignored `storage_data/` directory. To use the existing GCS bucket locally, set `GCS_BUCKET` in `.env` and authenticate with Google Application Default Credentials (for example, run `gcloud auth application-default login`). Grant the local identity object read/write access to the bucket; do not commit a service-account JSON key. The app stores objects using these keys:

```text
raw_pdfs/{doc_id}/{filename}
extracted_images/{doc_id}/p{page}_fig{number}.{ext}
vector_index/{doc_id}.json.gz
```

`doc_id` is the SHA-256 hash of the uploaded PDF. A later upload of the same PDF can restore its saved embeddings into the local Chroma collection. In Cloud Run, the service identity needs bucket object read/write access. The app currently calls the Gemini Developer API with `GEMINI_API_KEY`; it does not use Vertex AI service-account authentication for model calls.

The UI includes a rendered page preview, figure previews beside source passages, streamed answers, and checks that citations refer to retrieved passages and correct page numbers. A PDF preview is rendered locally by PyMuPDF from the stored PDF bytes.

## Deploy to Google Cloud Run

The included `Dockerfile` packages the app for Cloud Run. Enable the Cloud Run, Cloud Build, and Secret Manager APIs in your Google Cloud project, create a Secret Manager secret named `gemini-api-key` containing the Gemini key, and grant the Cloud Run service identity permission to access that secret. Then deploy from this directory:

```bash
gcloud run deploy paperpilot \
  --source . \
  --region YOUR_REGION \
  --set-secrets GEMINI_API_KEY=gemini-api-key:latest \
  --set-env-vars GCS_BUCKET=YOUR_BUCKET_NAME \
  --memory 2Gi \
  --timeout 3600 \
  --no-allow-unauthenticated
```

Use `--allow-unauthenticated` only if the app should be publicly accessible. Set `GEMINI_MODEL` as a Cloud Run environment variable if you want to use a different Gemini model.

**Storage note:** ChromaDB runs on the container filesystem. The app also saves each document's embeddings and metadata as a compressed snapshot in GCS, so a re-upload of the same PDF can restore its index after an instance restart. A live Chroma database is not shared between instances; users may need to upload the PDF again to restore it on a new instance. For multi-user production use, add authentication and document access controls.

---

## 🚀 Key Features

- 📄 **Research Paper Upload**
  - Upload a research paper in PDF format.
  - The uploaded paper becomes the primary knowledge source for the chatbot.

- 💬 **Conversational Q&A**
  - Ask natural-language questions about the uploaded research paper.
  - Interact with the document through a conversational interface.

- 🔎 **Retrieval-Augmented Generation (RAG)**
  - Relevant sections of the research paper are retrieved for every user query.
  - Retrieved content is provided to the LLM as contextual evidence before generating an answer.

- 🛡️ **Hallucination Mitigation**
  - Generated responses are grounded against the original research paper.
  - Retrieved evidence is used to verify whether generated claims are supported by the source document.
  - If sufficient evidence cannot be found, the system can avoid presenting unsupported information as fact.

- 📝 **Text Extraction**
  - Extracts textual content from research papers.
  - Processes extracted text for downstream chunking and retrieval.

- 📊 **Table Extraction**
  - Extracts tables and structured information from research papers.
  - Preserves important experimental results and numerical information that may otherwise be lost during plain-text extraction.

- 🖼️ **Image and Figure Extraction**
  - Extracts relevant images and figures from the PDF.
  - Processes visual information for inclusion in the searchable document representation.

- 🤖 **Image Captioning**
  - Generates semantic descriptions of extracted images and figures.
  - Makes visual information accessible to the retrieval pipeline.

- 🧠 **Multimodal Document Understanding**
  - Combines information from text, tables, and figures.
  - Allows the chatbot to answer questions based on different information types present in a research paper.

- ⚡ **Vector Search with ChromaDB**
  - Uses ChromaDB as the local vector database.
  - Converts document content into embeddings for semantic retrieval.

---

# 🏗️ System Architecture

```text
                         ┌─────────────────────┐
                         │    Research Paper   │
                         │        PDF          │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │   Document Parser   │
                         └──────────┬──────────┘
                                    │
                  ┌─────────────────┼─────────────────┐
                  │                 │                 │
                  ▼                 ▼                 ▼
          ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
          │     Text     │  │    Tables    │  │    Images    │
          │  Extraction  │  │  Extraction  │  │  Extraction  │
          └──────┬───────┘  └──────┬───────┘  └──────┬───────┘
                 │                 │                 │
                 │                 │                 ▼
                 │                 │        ┌─────────────────┐
                 │                 │        │ Image Captioning│
                 │                 │        └────────┬────────┘
                 │                 │                 │
                 └─────────────────┼─────────────────┘
                                   │
                                   ▼
                         ┌─────────────────────┐
                         │ Document Chunking   │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Embedding Generation│
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │      ChromaDB       │
                         │   Vector Database   │
                         └──────────┬──────────┘
                                    │
                                    │
                          User Question
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │  Query Processing   │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │   Semantic Search   │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Retrieved Evidence  │
                         │   from Research     │
                         │        Paper        │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │        LLM          │
                         │ Answer Generation   │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Evidence / Response │
                         │    Verification     │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Grounded Final      │
                         │      Answer         │
                         └─────────────────────┘
