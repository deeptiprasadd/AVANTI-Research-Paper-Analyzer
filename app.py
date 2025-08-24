# app.py — Research Paper Analyzer (fixed theme/patterns/custom color + rich comments)
# ------------------------------------------------------
# NOTE: This file focuses on three things you asked for:
# 1) Proper theme & background that doesn't wash the whole UI into "light".
# 2) Patterns (dots/grid/stripes) that actually show up on the page container.
# 3) Custom background color that always works, independent of preset.
# Plus: Very easy English comments after every variable/loop/step.

import re  # for regex based cleaning, finding sections, tokenization
import io  # for in-memory byte streams (PDF bytes -> BytesIO)
import time  # to measure processing time for metrics
from typing import Tuple, Dict, List  # for clear type hints

import requests  # to call arXiv API
import feedparser  # to parse arXiv Atom feeds
import streamlit as st  # UI framework

# Text features and similarity for simple extractive summarization
from sklearn.feature_extraction.text import TfidfVectorizer  # TF-IDF vectorizer
from sklearn.metrics.pairwise import cosine_similarity  # for MMR diversity

# ======================= PDF backends =======================
PDF_BACKEND = None  # holds backend name string (for diagnostics)
PdfReader = None  # will hold a PdfReader class if available
pdfplumber_available = False  # flag to know if pdfplumber is present

try:
    import pdfplumber  # very good for PDF text extraction
    pdfplumber_available = True  # mark available
    PDF_BACKEND = "pdfplumber"  # set backend name
except Exception:
    pass  # if pdfplumber import fails, we try pypdf / PyPDF2 below

if not pdfplumber_available:  # if first choice failed
    try:
        from pypdf import PdfReader as _PdfReader  # modern fork of PyPDF2
        PdfReader = _PdfReader  # assign reader class
        PDF_BACKEND = "pypdf"  # mark backend
    except Exception:
        try:
            from PyPDF2 import PdfReader as _PdfReader  # fallback
            PdfReader = _PdfReader  # assign reader class
            PDF_BACKEND = "PyPDF2"  # mark backend
        except Exception:
            PDF_BACKEND = None  # no backend available

# ======================= Page config (microscope favicon) =======================
st.set_page_config(  # Streamlit page meta
    page_title="AVANTI -Research Paper Analyzer",  # title for the tab
    page_icon="🔬",  # small emoji icon (microscope)
    layout="wide",  # use wide layout for better reading
)

# ----------------------- Theme utilities -----------------------
def _luminance(hex_color: str) -> float:
    """
    Compute relative luminance of a hex color.
    Easy English: This tells us how bright the color is.
    We use this to choose safe text and pattern contrast.
    """
    hc = hex_color.lstrip("#")  # remove leading '#'
    if len(hc) == 3:  # handle short hex like #abc
        hc = "".join([c * 2 for c in hc])  # expand to 6 chars
    # convert hex pairs to 0..1 floats
    r = int(hc[0:2], 16) / 255.0
    g = int(hc[2:4], 16) / 255.0
    b = int(hc[4:6], 16) / 255.0

    # gamma correction to linear space
    def lin(c):
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    R, G, B = lin(r), lin(g), lin(b)  # linear values
    return 0.2126 * R + 0.7152 * G + 0.0722 * B  # standard luminance formula


def build_background_css(bg_hex: str, pattern: str) -> str:
    """
    Create CSS for background color + chosen subtle pattern.
    Easy English: We paint your page with a solid color first,
    then put a soft pattern layer on top. Pattern uses safe contrast
    based on how bright the background is.
    """
    base = bg_hex  # your chosen background
    # decide pattern overlay color: light bg -> dark overlay, dark bg -> light overlay
    is_light = _luminance(bg_hex) > 0.5  # True if bright
    # pattern color with transparency (alpha via rgba) for subtle effect
    # We keep it subtle so text is always readable.
    pat = "rgba(0,0,0,0.12)" if is_light else "rgba(255,255,255,0.12)"

    # Different CSS patterns using multiple background-image layers
    if pattern == "Solid":
        # simple: only solid color, no pattern image
        return f"background-color: {base};"
    elif pattern == "Dots":
        # small radial circles as dots
        return (
            f"background-color: {base};"
            f"background-image: radial-gradient({pat} 1px, transparent 1px);"
            "background-size: 18px 18px;"
            "background-attachment: fixed;"
        )
    elif pattern == "Grid":
        # soft grid using two perpendicular linear gradients
        return (
            f"background-color: {base};"
            f"background-image: "
            f"linear-gradient({pat} 1px, transparent 1px),"
            f"linear-gradient(90deg, {pat} 1px, transparent 1px);"
            "background-size: 22px 22px;"
            "background-attachment: fixed;"
        )
    elif pattern == "Diagonal Stripes":
        # repeating diagonal stripes
        return (
            f"background-color: {base};"
            f"background-image: repeating-linear-gradient(45deg, {pat} 0, {pat} 8px, transparent 8px, transparent 24px);"
            "background-attachment: fixed;"
        )
    else:
        # fallback: only base color
        return f"background-color: {base};"


def apply_theme(bg_hex: str, pattern: str, font_pair: str, use_cards: bool):
    """
    Inject CSS for theme, fonts, and layout only on the main app container.
    Easy English: We style the app background safely, pick readable fonts,
    and keep content centered. We do not flood every widget with a white box.
    """
    # decide main text colors based on background brightness
    light_bg = _luminance(bg_hex) > 0.5  # True if bright background
    text_color = "#101418" if light_bg else "#f4f6f8"  # main text color
    # secondary text color (slightly softer than main)
    subtext_color = "#2c3238" if light_bg else "#dfe3e8"
    # optional card bg (slightly transparent white on dark bg or very light on light bg)
    card_bg = "rgba(255,255,255,0.08)" if not light_bg else "rgba(0,0,0,0.03)"

    # font selection map (you can extend this)
    # Easy English: Heading font and body font pairs.
    if font_pair == "Inter + Playfair":
        heading_font = "Playfair Display"
        body_font = "Inter"
        google_fonts = "family=Inter:wght@400;500;600&family=Playfair+Display:wght@600;700"
    elif font_pair == "Poppins + Merriweather":
        heading_font = "Merriweather"
        body_font = "Poppins"
        google_fonts = "family=Poppins:wght@400;500;600&family=Merriweather:wght@700;800"
    else:  # "System Sans"
        heading_font = "system-ui"
        body_font = "system-ui"
        google_fonts = ""  # no external fonts

    # build base background CSS (color + pattern)
    bg_css = build_background_css(bg_hex, pattern)  # final background CSS string

    # decide whether to style common content blocks as "cards"
    card_css = (
        f"""
        /* Optional subtle cards for readability */
        .stMarkdown, .stTextArea, .stDataFrame, .stCode, .stAlert, .stMetric, .stTabs {{
            background: {card_bg};
            border-radius: 14px;
            padding: 6px 10px;
        }}
        """
        if use_cards
        else ""
    )

    # inject complete style
    st.markdown(
        f"""
        <style>
        /* Google Fonts (optional) */
        {"@import url('https://fonts.googleapis.com/css2?" + google_fonts + "&display=swap');" if google_fonts else ""}

        /* Apply background ONLY to app container to avoid washing everything */
        [data-testid="stAppViewContainer"] {{
            {bg_css}
        }}

        /* Keep content centered and readable width */
        .block-container {{
            max-width: 950px;  /* comfortable reading width */
            margin-left: auto;
            margin-right: auto;
        }}

        /* Base text colors for whole app area */
        [data-testid="stAppViewContainer"], [data-testid="stSidebar"] * {{
            color: {text_color};
        }}

        /* Typography for headings */
        h1, h2, h3, h4, h5, h6 {{
            font-family: "{heading_font}", serif;
            color: {text_color};
            text-align: center;
            font-weight: 700;
            letter-spacing: 0.2px;
            margin-top: 0.3rem;
        }}

        /* Body text, inputs, buttons */
        .stMarkdown, .stText, p, li, label, .stButton button, .stSelectbox, .stSlider, .stTextInput, .stRadio, .stCheckbox {{
            font-family: "{body_font}", -apple-system, Segoe UI, Roboto, sans-serif;
            color: {text_color};
        }}
        p {{ line-height: 1.6; }}

        /* Tabs centered */
        div[data-baseweb="tab-list"] {{ justify-content: center; }}

        /* Sidebar contrast tweaks */
        section[data-testid="stSidebar"] {{
            backdrop-filter: blur(3px);
        }}

        {card_css}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ======================= Static title =======================
st.markdown("<h1>AVANTI</h1><h3 style='text-align:center; color:gray;'>Research Paper Analyzer</h3>", unsafe_allow_html=True)

# ======================= Small helpers =======================
def metric_row(cols, items):
    """
    Show a row of metrics.
    cols: tuple/list of st.columns
    items: list of (label, value) pairs
    """
    for col, (label, value) in zip(cols, items):  # loop through each metric entry
        col.metric(label, value)  # show Streamlit metric


def normalize_spaces(s: str) -> str:
    """Remove extra spaces and tidy newlines. Easy English: Make text clean."""
    return re.sub(r"[ \t]+", " ", re.sub(r"\s+\n", "\n", s)).strip()


def join_pages(pages: List[str]) -> str:
    """Join page texts with form-feed separators. Easy English: Keep page splits."""
    return "\f".join(p if p else "" for p in pages)  # keep empty pages as empty strings


# ======================= Read PDF =======================
@st.cache_data(show_spinner=False)
def read_pdf_text(file_bytes: bytes, max_pages: int = None) -> Tuple[str, List[str], Dict]:
    """
    Read text from PDF using available backend.
    Returns: (all_text_joined, list_of_page_texts, meta_info)
    """
    bio = io.BytesIO(file_bytes)  # wrap bytes for readers
    texts: List[str] = []  # store text of each page here
    total_pages = 0  # total pages count

    if PDF_BACKEND == "pdfplumber":  # prefer pdfplumber if available
        with pdfplumber.open(bio) as pdf:  # open PDF
            total_pages = len(pdf.pages)  # count pages
            pages_to_read = pdf.pages if not max_pages else pdf.pages[:max_pages]  # limit pages
            for page in pages_to_read:  # loop over pages
                texts.append(page.extract_text() or "")  # safe extract (empty if None)
    elif PdfReader:  # fallback readers
        reader = PdfReader(bio)  # create reader
        total_pages = len(reader.pages)  # count pages
        for i, page in enumerate(reader.pages):  # iterate with index
            if max_pages and i >= max_pages:  # stop if reached limit
                break
            try:
                texts.append(page.extract_text() or "")  # push page text
            except Exception:
                texts.append("")  # push empty if extraction fails
    else:
        # no backend available
        return "", [], {"total_pages": 0, "pages_processed": 0}

    # join texts and create meta
    return join_pages(texts), texts, {"total_pages": total_pages, "pages_processed": len(texts)}


# ======================= Span: Abstract → References =======================
ABSTRACT_PAT = re.compile(r"(?im)^\s*(?:\d+\.\s*)?(ABSTRACT|Abstract)\s*:?\s*$")  # to detect Abstract header
REFS_PAT = re.compile(r"(?im)^\s*(References|Bibliography)\s*:?\s*$")  # to detect References header

def locate_abstract_to_references(all_text: str) -> Tuple[int, int]:
    """
    Find line indexes for Abstract start and References start.
    Easy English: We try to cut only the main core of the paper.
    """
    lines = all_text.splitlines()  # split to lines
    abs_idx, refs_idx = None, None  # will keep positions

    for i, ln in enumerate(lines):  # scan for headings
        if abs_idx is None and ABSTRACT_PAT.match(ln.strip()):
            abs_idx = i  # mark abstract line
        if REFS_PAT.match(ln.strip()):
            refs_idx = i  # mark references line
            break  # stop after we find references

    if abs_idx is None:  # if no "Abstract", pick first non-empty line
        for i, ln in enumerate(lines):
            if ln.strip():
                abs_idx = i
                break

    if refs_idx is None:
        refs_idx = len(lines)  # set to end if no references header

    return abs_idx or 0, refs_idx  # return safe positions


def slice_abstract_to_refs(all_text: str) -> str:
    """
    Get text only from Abstract to References (excluded).
    Easy English: This is the core we summarize.
    """
    lines = all_text.splitlines()  # lines list
    i_abs, i_refs = locate_abstract_to_references(all_text)  # get positions
    core = "\n".join(lines[i_abs:i_refs])  # slice the core
    core = re.sub(REFS_PAT, "", core)  # remove possible header text inside
    return core.strip()  # clean edges


# ======================= Title + Authors + Affiliations =======================
_NOISE_HDR = re.compile(
    r"(ISSN|ISBN|DOI|doi\.org|arXiv|www\.|http|Received|Revised|Accepted|Published|"
    r"International\s+Journal|Conference|Proceedings|Volume|Vol\.|Issue|Pages?)",
    re.I,
)  # lines with this are likely not title/author
AFFIL_KEYWORDS = (
    "University","Institute","Department","College","Lab","Laboratory","School","Centre","Center",
    "Academy","CSIR","IIT","NIT","AIIMS","Hospital","Clinic","Research","Faculty","Campus",
    "Delhi","India","USA","UK","Iran","Germany","France","Korea","China"
)  # hints for affiliations
EMAIL_PAT = re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.I)  # find email
NAME_TOKEN = r"[A-Z][A-Za-z\-']+"  # pattern for name token (capitalized)
NAME_HINT = re.compile(rf"({NAME_TOKEN}(?:\s+(?:{NAME_TOKEN}|[A-Z]\.))+)")
SUP = r"(?:\d+|[¹²³⁴⁵⁶⁷⁸⁹])"  # superscripts or numbers for affiliations
SUP_AT = re.compile(rf"{NAME_TOKEN}(?:\s+(?:{NAME_TOKEN}|[A-Z]\.))+{SUP}?")  # name with optional sup

def _clean_line(ln: str) -> str:
    """Trim quotes and multiple spaces."""
    ln = re.sub(r"\s+", " ", ln).strip()
    return ln.strip(' "\'“”‚‟')

def _is_noise_line(ln: str) -> bool:
    """Remove lines that look like headers/footers or are too short."""
    if not ln or len(ln) < 3:
        return True
    if _NOISE_HDR.search(ln):
        return True
    return False

def _cap_ratio(text: str) -> float:
    """Share of tokens that look capitalized. Helps estimate title-like lines."""
    toks = re.findall(r"[A-Za-z][A-Za-z\-']*", text)
    if not toks:
        return 0.0
    caps = sum(1 for t in toks if t[0].isupper() or t.isupper())
    return caps / max(1, len(toks))

def extract_title_authors_affils(page_texts: List[str]) -> Tuple[str, List[str], List[str]]:
    """
    Heuristics to guess Title, Authors and Affiliations from first pages.
    Easy English: We read the first page or two and try to pull header info.
    """
    header_lines: List[str] = []  # keep top-of-paper lines here first
    for pg in page_texts[:2]:  # check first up to 2 pages
        stop = False  # will stop when Abstract appears
        for ln in pg.splitlines():  # loop through lines on page
            if ABSTRACT_PAT.match(ln.strip()):  # if "Abstract" found
                stop = True  # mark stop
                break  # break inner loop
            header_lines.append(_clean_line(ln))  # keep cleaned line
        if stop:  # if abstract found on this page
            break  # break outer loop

    # remove empty and noisy lines
    lines = [ln for ln in header_lines if ln and not _is_noise_line(ln)]

    # find best title candidate by scoring small blocks of lines
    best_text, best_score, best_span = "", -1e9, (0, 1)  # track best candidate
    for i in range(len(lines)):  # loop over all starting positions
        for w in (1, 2, 3):  # try 1, 2, or 3 lines combined
            if i + w > len(lines):
                break
            block = lines[i : i + w]  # lines block
            text = " ".join(block)  # join to one string
            words = len(text.split())  # word count
            if words < 4 or words > 26:
                continue  # skip too short/long
            if any(k in text for k in AFFIL_KEYWORDS):
                continue  # skip if contains affiliation hints
            cap = _cap_ratio(text)  # capitalization score
            kw = 1 if re.search(r"\b(using|analysis|approach|framework|method|survey|prediction|"
                                r"classification|detection|design|review)\b", text, re.I) else 0  # keyword bonus
            len_bonus = 2 if 6 <= words <= 18 else 0  # length bonus
            bad_end = 1.5 if text.endswith((".", "…")) else 0  # penalty if ends like a sentence
            score = 2.0 * cap + kw + len_bonus - bad_end  # final score
            if score > best_score:  # keep best
                best_text, best_score, best_span = text, score, (i, w)

    title = (best_text or (lines[0] if lines else "")).strip(' "\'“”')  # final title guess

    start = best_span[0] + best_span[1] if best_text else 1  # where authors likely start
    candidate_zone = lines[start : start + 20]  # scan next lines for authors/affils

    raw_author_lines: List[str] = []  # possible lines with author names
    raw_affil_lines: List[str] = []  # possible lines with affiliations

    for ln in candidate_zone:  # scan each candidate line
        has_email = EMAIL_PAT.search(ln) is not None  # email present
        has_affil_kw = any(k.lower() in ln.lower() for k in AFFIL_KEYWORDS)  # affiliation words present
        has_name = NAME_HINT.search(ln) is not None or SUP_AT.search(ln) is not None  # name-ish structure

        if has_name and (not has_affil_kw or ln.count(",") >= 1) and len(ln.split()) <= 20:
            raw_author_lines.append(ln)  # likely author line
        if has_affil_kw or re.match(rf"^{SUP}\s", ln) or re.match(r"^\d+\s", ln):
            raw_affil_lines.append(ln)  # likely affiliation line
        if has_email and not has_affil_kw and not has_name:
            continue  # emails alone are not useful as name/affil lines

    # extract authors by splitting on commas and 'and'
    authors: List[str] = []  # final authors
    for ln in raw_author_lines:
        ln = EMAIL_PAT.sub("", ln)  # remove emails
        ln = re.sub(r"[¹²³⁴⁵⁶⁷⁸⁹]|\b\d+\b", "", ln)  # remove superscripts/numbers
        parts = re.split(r"\s+and\s+|,\s*", ln)  # split possible multi-author line
        for p in parts:
            p = p.strip(" ,;:·-")  # clean edges
            # accept if pattern looks like real name (2-5 tokens)
            if NAME_HINT.fullmatch(p) or re.match(rf"^{NAME_TOKEN}(?:\s+(?:{NAME_TOKEN}|[A-Z]\.))+$", p):
                if 2 <= len(p.split()) <= 5 and p not in authors:
                    authors.append(p)  # push unique
        if len(authors) >= 10:
            break  # safety limit

    # extract affiliations from candidate lines
    affils: List[str] = []  # final affiliations
    for ln in raw_affil_lines:
        ln = EMAIL_PAT.sub("", ln)  # remove emails
        ln = re.sub(r"\s{2,}", " ", ln).strip(" ,;:-")  # normalize spaces
        if len(ln.split()) >= 3 and ln not in affils:
            affils.append(ln)  # push unique
        if len(affils) >= 10:
            break  # safety limit

    # fallback: scan candidate zone for any line that looks like affiliation
    if not affils:
        for ln in candidate_zone:
            if any(k.lower() in ln.lower() for k in AFFIL_KEYWORDS) and len(ln.split()) >= 3:
                ln = EMAIL_PAT.sub("", ln).strip(" ,;:-")
                if ln not in affils:
                    affils.append(ln)

    authors = [a.strip(' "\'“”') for a in authors]  # final clean
    affils = [a.strip(' "\'“”') for a in affils]  # final clean
    return title, authors, affils  # return header info


# ======================= Cleaning & sentence splitting =======================
def strip_citations_inline(s: str) -> str:
    """
    Remove [1], (Author, 2020), Fig. 1, etc. from a sentence.
    Easy English: Keep only the main content words.
    """
    s = re.sub(r"\[[0-9,\s\-;]{1,20}\]", " ", s)  # [1], [2,3], [1-5]
    s = re.sub(r"\([A-Z][A-Za-z\-]+(?:\s+et\s+al\.)?,?\s*\d{4}[a-z]?\)", " ", s)  # (Smith 2020)
    s = re.sub(r"\(\d{4}[a-z]?\)", " ", s)  # (2020)
    s = re.sub(r"\b(Figure|Fig\.|Table|Eq\.?|Section)\s*\d+[a-z]?\b", " ", s, flags=re.I)  # Fig./Table refs
    s = re.sub(r"\b(see|refer to)\s+(Figure|Table|Section)\b.*?[\.]", ". ", s, flags=re.I)  # see Figure...
    s = re.sub(r"\s{2,}", " ", s)  # squeeze spaces
    return s.strip()  # clean ends


def is_reference_like(sent: str) -> bool:
    """Return True if the sentence looks like a reference entry."""
    s = sent.lower()
    if any(x in s for x in ["doi.org", "http://", "https://", "arxiv", "vol.", "no.", "pp.", "isbn"]):
        return True
    if re.search(r"\b\d{4}\b", s) and re.search(r"[A-Z][a-z]+,?\s+[A-Z]\.", sent):
        return True
    if re.fullmatch(r"[A-Za-z,\s&\-]+(?:\(\d{4}[a-z]?\))[,;\sA-Za-z\(\)\d\-]*", sent.strip()):
        return True
    return False


def sent_tokenize(text: str) -> List[str]:
    """
    Simple sentence split based on punctuation and capitalization.
    Easy English: Cut paragraph into sentences.
    """
    text = text.replace("\r", " ")  # normalize CR
    text = re.sub(r"-\s*\n\s*", "", text)  # fix hyphen line-breaks
    text = re.sub(r"\s*\n\s*", " ", text)  # join wrapped lines
    sents = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text)  # split after .!? if next is capital/number
    return [normalize_spaces(s) for s in sents if len(s.strip()) > 25]  # filter very short


# ======================= Summarizer (TF-IDF + MMR) =======================
def tfidf_mmr_summary(text: str, target_words: int = 900, diversity: float = 0.45) -> str:
    """
    Build an extractive summary using TF-IDF scoring + MMR for diversity.
    Easy English: Pick important sentences, avoid repeats, reach target size.
    """
    sents = sent_tokenize(text)  # list of sentences
    if not sents:
        return ""  # nothing to summarize

    cleaned: List[str] = []  # filtered sentences after removing citations
    for s in sents:  # go through each sentence
        s2 = strip_citations_inline(s)  # remove citation noise
        if not is_reference_like(s2):  # skip reference-style lines
            cleaned.append(s2)

    if not cleaned:
        cleaned = sents  # fallback to original sentences

    if len(cleaned) > 1500:
        cleaned = cleaned[:1500]  # safety cap

    vec = TfidfVectorizer(stop_words="english", max_df=0.9, ngram_range=(1, 2))  # vectorizer config
    X = vec.fit_transform(cleaned)  # sentence-term matrix

    sal = (X.power(2).sum(axis=1)).A1  # sentence salience as TF-IDF energy
    ranked = sal.argsort()[::-1]  # indices from most to least salient
    need = max(6, min(len(cleaned) // 2, target_words // 20))  # rough number of sentences to collect

    selected: List[int] = []  # chosen sentence indices
    selected_vecs: List = []  # chosen sentence vectors
    for idx in ranked:  # loop in order of importance
        if len(selected) >= need:  # stop when enough sentences are chosen
            break
        cand = X[idx]  # candidate vector

        if not selected_vecs:  # first sentence always chosen
            selected.append(idx)
            selected_vecs.append(cand)
            continue

        # compute redundancy (max cosine sim with selected ones)
        red = max((cosine_similarity(cand, v)[0, 0] for v in selected_vecs), default=0.0)
        mmr = (1 - diversity) * sal[idx] - diversity * red  # MMR score combines importance & novelty
        if red < 0.7 or mmr > 0:  # accept if not too similar or still valuable
            selected.append(idx)
            selected_vecs.append(cand)

    selected = sorted(selected)  # keep original order for readability
    summary = " ".join(cleaned[i] for i in selected)  # glue sentences
    words = summary.split()  # list of words
    if len(words) > target_words:
        summary = " ".join(words[:target_words]) + "..."  # trim softly

    return summary  # final text


def to_paragraphs(text: str, sentences_per_para: int = 3) -> str:
    """
    Split summary into paragraphs of N sentences.
    Easy English: Make the summary look neat.
    """
    sents = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text.strip())  # re-split safely
    sents = [s.strip() for s in sents if s.strip()]  # clean
    paras: List[str] = []  # list of paragraph strings
    for i in range(0, len(sents), sentences_per_para):  # step in chunks
        paras.append(" ".join(sents[i : i + sentences_per_para]))  # join chunk
    return "\n\n".join(paras)  # join with blank line


# ======================= References =======================
def extract_references(all_pages: List[str]) -> List[str]:
    """
    Heuristic to read references from the last pages.
    Easy English: We try common list formats like [1] or 1) etc.
    """
    block = "\n".join(all_pages[-3:]) if all_pages else ""  # last 3 pages joined
    refs: List[str] = []  # output list

    # style like: [12] Author ... Title ...
    for m in re.finditer(r"(?ms)^\s*\[?(\d{{1,3})\]?\s+(.+?)(?=^\s*\[?\d{{1,3}}\]?\s+|$)", block):
        num, body = m.group(1), re.sub(r"\s+", " ", m.group(2)).strip()
        if len(body) > 20:
            refs.append(f"[{num}] {body}")

    # fallback: 1) or 1. prefixed lines
    if not refs:
        for m in re.finditer(r"(?ms)^\s*(\d{{1,3})[\.\)]\s+(.+?)(?=^\s*\d{{1,3}}[\.\)]\s+|$)", block):
            num, body = m.group(1), re.sub(r"\s+", " ", m.group(2)).strip()
            if len(body) > 20:
                refs.append(f"[{num}] {body}")

    # de-duplicate while keeping order
    out: List[str] = []  # final list
    seen = set()  # seen entries
    for r in refs:
        if r not in seen:
            out.append(r)
            seen.add(r)

    return out[:200]  # cap to 200 items for safety


# ======================= Sidebar (Settings + THEME SWITCHER) =======================
with st.sidebar:
    st.header("Settings")  # title for the sidebar
    st.caption(f"PDF backend: {PDF_BACKEND or 'None'}")  # show which PDF library is in use

    # -- Theme & Background Controls --
    st.subheader("Theme & Background")  # section title

    # preset theme picker (only provides a starting color)
    preset = st.selectbox(
        "Preset theme",  # label
        ["Light", "Midnight (Black)", "Ocean (Blue)", "Plum (Purple)", "Teal"],  # options
        index=0,  # default selection
    )

    # font pair selection: you can choose fonts you like
    font_pair = st.selectbox(
        "Font style",
        ["Inter + Playfair", "Poppins + Merriweather", "System Sans"],
        index=0,
    )

    # choose a custom color (works regardless of preset)
    custom_color = st.color_picker("Custom background color", "#ffffff")  # color hex

    # pick a pattern
    pattern = st.selectbox(
        "Pattern",
        ["Solid", "Dots", "Grid", "Diagonal Stripes"],
        index=0,
    )

    # small toggle: use subtle cards or not
    use_cards = st.checkbox(
        "Use subtle cards for sections",
        value=True,
        help="If ON, content blocks get a soft card background to improve readability.",
    )

    # map presets to base colors
    preset_map = {
        "Light": "#ffffff",
        "Midnight (Black)": "#0b0f14",
        "Ocean (Blue)": "#0a2540",
        "Plum (Purple)": "#2b0a3d",
        "Teal": "#0b3d3a",
    }

    # start from preset color
    chosen_bg = preset_map.get(preset, "#ffffff")  # initial background color

    # IMPORTANT FIX:
    # If user picks any color different from the preset default, ALWAYS apply that color.
    # Easy English: Your custom color wins over the preset.
    if custom_color and custom_color.lower() != chosen_bg.lower():
        chosen_bg = custom_color  # override with custom

    # apply CSS theme now using the chosen settings
    apply_theme(chosen_bg, pattern, font_pair, use_cards)

    # Existing controls for app logic
    max_pages = st.number_input("Max PDF pages to read", 1, 500, 40)  # how many pages to parse
    summary_len = st.slider("Summary target length (words)", 300, 2000, 900, step=50)  # target length
    search_limit = st.number_input("Max search results", 1, 20, 6)  # arXiv results count

    # helper info
    with st.expander("What do these mean?"):
        st.markdown(
            "- **Theme & Background**: Choose a preset, your own color, and a subtle pattern.\n"
            "- **Font style**: Pick the font pairing you prefer.\n"
            "- **Use subtle cards**: Turn small section cards ON/OFF.\n"
            "- **PDF backend**: Library used to extract text (pdfplumber / pypdf / PyPDF2).\n"
            "- **Max PDF pages to read**: Number of pages parsed from the start of the file.\n"
            "- **Summary target length (words)**: Desired size of the summary.\n"
            "- **Abstract→References**: Only this span is summarized; References are excluded and shown separately."
        )

# ======================= Tabs =======================
tab_search, tab_upload = st.tabs(["Search research papers by name", "Upload & summarize"])  # two main tabs

# ---------- Search ----------
def build_arxiv_pdf_url(entry_id: str = "", pdf_url: str = "") -> str:
    """
    Build a clean arXiv PDF URL from feed fields.
    Easy English: Make sure 'Open PDF' button points to correct link.
    """
    if pdf_url and "/pdf/" in pdf_url:
        slug = pdf_url.split("/pdf/")[-1].split("?")[0].replace(".pdf", "")
        return f"https://arxiv.org/pdf/{slug}.pdf"
    if entry_id and "/abs/" in entry_id:
        slug = entry_id.split("/abs/")[-1]
        return f"https://arxiv.org/pdf/{slug}.pdf"
    return pdf_url or entry_id  # fallback

with tab_search:
    st.subheader("Search your research paper")  # section title
    # text input to accept query
    query = st.text_input("Enter paper title or keywords", "e.g., autism spectrum disorder detection")

    # search button click
    if st.button("Search"):
        with st.spinner("Searching..."):  # show loading spinner
            url = f"http://export.arxiv.org/api/query?search_query=all:{query}&start=0&max_results={search_limit}"  # arXiv API url
            r = requests.get(url, timeout=12)  # GET request with timeout
            feed = feedparser.parse(r.text)  # parse Atom XML

        if not feed.entries:  # if nothing returned
            st.warning("No results.")  # show warning

        # loop through results
        for entry in feed.entries:
            title = entry.title  # paper title
            authors = ", ".join(a.name for a in getattr(entry, "authors", []))  # authors joined
            published = getattr(entry, "published", "")  # publish date
            abstract = getattr(entry, "summary", "").strip()  # paper abstract
            entry_id = getattr(entry, "id", "")  # abstract page URL
            pdf_url = build_arxiv_pdf_url(entry_id, getattr(entry, "pdf_url", ""))  # clean PDF URL

            # each result in an expander
            with st.expander(title):
                st.write(f"**Authors:** {authors}")
                st.write(f"**Published:** {published}")
                st.write(f"**Abstract:** {abstract}")
                st.link_button("Open PDF", pdf_url)  # button to open PDF
                st.link_button("Open Abstract Page", entry_id)  # button to open abstract page

# ---------- Upload ----------
with tab_upload:
    st.subheader("Upload a PDF")  # section title
    up = st.file_uploader("Choose a PDF", type=["pdf"])  # upload input

    # click to analyze
    if up and st.button("Analyze PDF"):
        t0 = time.time()  # start timer

        if not (PdfReader or pdfplumber_available):  # if no reader available
            st.error("No PDF parser available.")  # show error
        else:
            data = up.read()  # read file bytes
            all_text, page_texts, meta = read_pdf_text(data, max_pages=max_pages)  # extract text

            paper_title, authors, affils = extract_title_authors_affils(page_texts)  # get header info
            core_text = slice_abstract_to_refs(all_text)  # slice only core

            with st.spinner("Summarizing..."):  # show spinner
                raw_summary = tfidf_mmr_summary(core_text, target_words=summary_len, diversity=0.45)  # build summary
                pretty_summary = to_paragraphs(raw_summary, sentences_per_para=3)  # format for reading

            refs = extract_references(page_texts)  # pull references

            elapsed = time.time() - t0  # total seconds
            total_words = len(all_text.split())  # full document word count
            core_words = len(core_text.split())  # words inside core span

            # show metrics in 4 columns
            m1, m2, m3, m4 = st.columns(4)
            metric_row(
                (m1, m2, m3, m4),
                [
                    ("Pages processed", f"{meta['pages_processed']}/{meta['total_pages']}"),
                    ("Total words (document)", f"{total_words:,}"),
                    ("Words analyzed (Abstract→References)", f"{core_words:,}"),
                    ("Time (s)", f"{elapsed:.1f}"),
                ],
            )

            # header area
            st.subheader("Paper header")
            if paper_title:
                st.markdown(f"**Title:** {paper_title}")  # show title
            if authors:
                st.markdown("**Authors:**")  # label
                for a in authors:  # list authors
                    st.markdown(f"- {a}")
            else:
                st.caption("Authors not confidently detected.")  # fallback
            if affils:
                st.markdown("**Affiliations (colleges / departments):**")  # label
                for a in affils:  # list affiliations
                    st.markdown(f"- {EMAIL_PAT.sub('', a).strip()}")  # remove any emails
            else:
                st.caption("Affiliations not confidently detected.")  # fallback

            # summary area
            st.subheader("Summary")
            st.markdown(pretty_summary)  # show formatted summary
            st.download_button("Download Summary", raw_summary, "summary.txt")  # raw text download

            # extra tabs
            s1, s2, s3 = st.tabs(["Core extract (Abstract→References)", "References", "Keywords"])

            with s1:
                st.caption("Exact span used to build the summary.")  # help text
                st.text(core_text[:120000] if core_text else "No Abstract→References span detected.")  # show text

            with s2:
                if refs:
                    for r in refs:  # list each reference
                        st.write(r)
                else:
                    st.info("No references detected on the last pages.")  # fallback

            with s3:
                # Very simple keyword extraction by frequency (stopwords removed)
                cleaned = re.sub(r"[^a-zA-Z0-9\s-]", " ", core_text.lower())  # keep letters/numbers
                words = re.findall(r"[a-z]{4,}", cleaned)  # words with len >=4
                stop = {
                    "with","this","that","from","were","have","which","when","where","your","their","about","into","also",
                    "than","then","there","such","these","been","because","very","through","while","among","more","only",
                    "each","some","will","many","they","between","using","based","data","results","analysis","paper",
                    "study","chapter","used","abstract","introduction","conclusion","method","methods","methodology"
                }  # simple stoplist

                freq: Dict[str, int] = {}  # word -> count
                for w in words:  # loop through each word
                    if w not in stop:  # skip stopwords
                        freq[w] = freq.get(w, 0) + 1  # count

                topk = [w for w, _ in sorted(freq.items(), key=lambda x: -x[1])[:15]]  # top 15 words
                st.write(", ".join(topk) if topk else "—")  # show or dash
