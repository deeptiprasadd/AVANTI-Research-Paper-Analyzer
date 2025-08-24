AVANTI – Research Paper Analyzer

Full Guide


---

Part 1: How to Use AVANTI

Step 1: Launch the App

Run the Streamlit app:

In Terminal/Command Prompt: streamlit run app.py

Your browser will open - https://avanti-research-paper-analyzer2025.streamlit.app/



Step 2: Choose a Mode

You will see two tabs:

Search research papers by name

Upload & summarize



Step 3A: Search Research Papers (arXiv)

In Search research papers by name tab:

1. Enter title or keywords (e.g., “autism spectrum disorder detection”).


2. (Sidebar) Set Max search results.


3. Click Search.


4. For each result, open the expander to see:

Authors

Published date

Abstract

Buttons: Open PDF and Open Abstract Page





Step 3B: Upload & Summarize Your PDF

In Upload & summarize tab:

1. Click Choose a PDF and select your file.


2. (Sidebar) Adjust:

Max PDF pages to read

Summary target length (words)



3. Click Analyze PDF.


4. AVANTI will:

Read the PDF (first N pages)

Detect Title, Authors, Affiliations (best-effort)

Extract the core span (Abstract → References)

Generate a summary (TF‑IDF + MMR)

List detected references

Show keywords (simple frequency-based)



5. Use Download Summary to save the raw text summary.




Step 4: Read the Outputs

Metrics: pages processed, word counts, total time

Paper header: title, authors, affiliations

Summary: readable paragraphs

Tabs:

Core extract (Abstract→References) – exact span used

References – detected items from last pages

Keywords – top frequent terms (after stopword removal)




---

Part 2: Install and Run

Step 1: Requirements

Python 3.9+

Recommended packages (from the code):

streamlit, requests, feedparser

pdfplumber or pypdf/**PyPDF2` (auto‑fallback)

scikit-learn



Step 2: Install Dependencies

pip install streamlit requests feedparser pdfplumber pypdf PyPDF2 scikit-learn

Step 3: Start AVANTI

streamlit run app.py

Keep the terminal open while using the app.



---

Part 3: Workflow Diagram (Code Flow)

[User]
  ↓
Open AVANTI in browser
  ↓
Choose a tab: [Search] or [Upload]

[Sidebar Settings]
  - Max PDF pages to read
  - Summary target length (words)
  - Max search results (for arXiv)

[TABS]

(Search research papers by name)
  ↓
  text_input(query) → button("Search")
  ↓
  requests.get(arXiv API) → feedparser.parse()
  ↓
  build_arxiv_pdf_url(entry_id, pdf_url)
  ↓
  Expanders show metadata → link_button(Open PDF / Abstract)

(Upload & summarize)
  ↓
  file_uploader → button("Analyze PDF")
  ↓
  read_pdf_text(file_bytes, max_pages)
     ├─ pdfplumber (preferred)
     └─ pypdf / PyPDF2 (fallback)
  ↓
  extract_title_authors_affils(page_texts)
  ↓
  slice_abstract_to_refs(all_text)
  ↓
  tfidf_mmr_summary(core_text, target_words, diversity=0.45)
  ↓
  to_paragraphs(summary)
  ↓
  extract_references(page_texts)
  ↓
  Display metrics + header + summary + tabs
  ↓
  download_button("summary.txt")


---

Part 4: Function Descriptions

1) read_pdf_text(file_bytes, max_pages)

Purpose: Extracts text from the PDF using available backend.

Flow:

Tries pdfplumber → falls back to pypdf → falls back to PyPDF2.

Returns: (joined_text, page_texts_list, meta) where meta = {total_pages, pages_processed}.


Cache: @st.cache_data for faster re-runs.


2) locate_abstract_to_references(all_text)

Purpose: Finds line indices for Abstract start and References start.

Logic: Regex for Abstract/References; falls back sensibly if not found.


3) slice_abstract_to_refs(all_text)

Purpose: Returns only the core text between Abstract and References.

Use: Input to summarizer and keywords block.


4) extract_title_authors_affils(page_texts)

Purpose: Heuristically extracts paper title, author names, and affiliations from the first 1–2 pages.

Key heuristics:

Filters noise lines (ISSN/DOI/headers).

Scores candidate title blocks by capitalization, length, and keywords.

Parses likely author lines; detects affiliations by keywords and formats.



5) strip_citations_inline(sent) / is_reference_like(sent)

Purpose: Clean sentences by removing citation clutter and excluding reference-looking lines.


6) sent_tokenize(text)

Purpose: Splits long text into sentences using simple punctuation rules.

Note: Filters very short fragments.


7) tfidf_mmr_summary(text, target_words=900, diversity=0.45)

Purpose: Extractive summary using TF‑IDF salience and MMR for diversity.

Flow:

Vectorize sentences with TfidfVectorizer(ngram_range=(1,2), stop_words='english').

Rank by TF‑IDF energy; select with redundancy control (cosine similarity).

Trim to target words.



8) to_paragraphs(text, sentences_per_para=3)

Purpose: Formats the summary into readable paragraphs.


9) extract_references(all_pages)

Purpose: Heuristic extraction of references from last pages.

Patterns: [n] ... or n) ... / n. ...; de-duplicates and caps at 200.


10) build_arxiv_pdf_url(entry_id, pdf_url)

Purpose: Builds a clean arXiv PDF URL from feed fields.


11) UI Helpers

metric_row(cols, items): Renders KPI metrics in a row.

normalize_spaces, join_pages: General text cleanup utilities.



---

Part 5: Tips & Troubleshooting

No PDF parser available: Ensure at least one of pdfplumber, pypdf, or PyPDF2 is installed.

Title/Authors missing: Heuristics may fail on non-standard formats; summary will still work.

Long PDFs: Increase Max PDF pages to read gradually to balance speed and coverage.

Summary too short/long: Adjust Summary target length (words) in the sidebar.



---

Part 6: File Outputs

summary.txt: Raw, single‑paragraph summary (download via button).



---

Part 7: Labels in the App

Main Heading: AVANTI

Subheading: Research Paper Analyzer

Tabs: Search research papers by name, Upload & summarize



---

This guide mirrors the structure of your BNI‑Connect‑Scraper‑Extension documentation, adapted to the AVANTI app’s code and features.

