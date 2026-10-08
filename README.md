# H2S AI Builder Cup — Research Paper LLM Chatbot

An intelligent LLM-powered research assistant that allows users to upload a research paper in PDF format and interact with it through a conversational chatbot.

The system processes the uploaded research paper, extracts text, tables, and images, converts the extracted information into a searchable knowledge base, and uses Retrieval-Augmented Generation (RAG) to provide context-grounded answers.

The retrieved content from the original research paper is also used to verify generated responses, helping reduce unsupported claims and mitigate LLM hallucinations.

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
