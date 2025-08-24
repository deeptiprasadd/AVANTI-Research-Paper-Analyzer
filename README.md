# AVANTI: Research Paper Analyzer

Live Demo: [https://avanti-research-paper-analyzer2025.streamlit.app/](https://avanti-research-paper-analyzer2025.streamlit.app/)

---

## Overview

AVANTI is a Streamlit-based tool designed to simplify research paper exploration and analysis. It allows users to search for academic papers on arXiv or upload their own PDFs to receive high-quality extractive summaries, metadata, and reference extraction.

---

## Features

- **Search through arXiv** by paper title or keywords to view details including title, authors, abstract, and publication date.
- **Upload PDF files** to extract:
  - Text from the Abstract through References sections.
  - Detected Title, Authors, and Affiliations.
  - A clean summary generated using TF-IDF and Maximum Marginal Relevance (MMR).
  - Extracted References and a keyword list.

- **Interactive Dashboard** featuring:
  - Core extract (Abstract → References)
  - Reference list
  - Keyword extraction
  - File download of the summary

---

## Quick Start

### Search Papers via arXiv

1. Navigate to the **Search research papers by name** tab.
2. Enter keywords or a title (e.g., “autism spectrum disorder detection”).
3. Click **Search**.
4. Browse the results and use the **Open PDF** or **Open Abstract Page** links.

### Upload Your Own PDF

1. Navigate to the **Upload & summarize** tab.
2. Upload a valid PDF.
3. Adjust settings in the sidebar:
   - **Max PDF pages to read** (default: 40)
   - **Summary target length (words)** (default: 900)
   - **Max search results** (default: 6)
4. Click **Analyze PDF** to process the file.
5. Receive:
   - Title, authors, and affiliations
   - Extractive summary
   - Metrics (word counts, processing time)
   - Downloadable summary file
   - Extra tabs: Core extract, References, Keywords

---

## Technical Workflow

[User Action]
↓
Select Path → (ArXiv Search) OR (PDF Upload)
↓
[If PDF Upload]
- read_pdf_text() → Extract pages
- extract_title_authors_affils() → Detect header information
- slice_abstract_to_refs() → Focus on core content
↓
tfidf_mmr_summary() → Curate summary with diversity
to_paragraphs() → Organize summary into readable format
↓
extract_references() → Gather bibliographic entries
↓
[User Interface]
- Display metrics
- Render summary and additional tabs
- Provide summary download

  
## Function Reference

- **read_pdf_text(file_bytes, max_pages)**  
  Extracts text from PDF (supports pdfplumber, pypdf, PyPDF2).

- **extract_title_authors_affils(page_texts)**  
  Auto-detects title, authors, and their affiliations.

- **slice_abstract_to_refs(all_text)**  
  Extracts content from Abstract to References.

- **tfidf_mmr_summary(text, target_words, diversity)**  
  Generates an extractive summary using TF-IDF and MMR to ensure diversity.

- **to_paragraphs(text, sentences_per_para)**  
  Formats summary into paragraphs of specified length.

- **extract_references(all_pages)**  
  Identifies references in the last pages in common formats.

---

## Installation Instructions

1. Clone the repository:

   ```
   git clone <repository_url>
   cd <repository_directory>
2. Install dependencies:
- pip install -r requirements.txt
- streamlit run app.py

3. Dependencies
- Python ≥ 3.8
- Streamlit
- scikit-learn
- feedparser
- requests
- pdfplumber, pypdf, or PyPDF2 (for PDF parsing)

---

License & Attribution
This project is released under the MIT License.
Contributions and feedback are welcome.

