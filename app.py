import streamlit as st
import requests
import json
import io
import sys
import subprocess
import time

# --- 1. SETUP & UTILS ---

# This function checks if you have the necessary tools installed.
# If a tool is missing, it installs it automatically so the app doesn't crash.
def install_basics():
    required = ["pdfplumber", "pypdf", "requests", "streamlit", "feedparser"]
    for lib in required:
        try:
            __import__(lib)
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", lib])

install_basics()
import feedparser

# --- 2. CONFIGURATION ---
st.set_page_config(
    page_title="AVANTI Research",
    page_icon="📑",
    layout="wide",
    initial_sidebar_state="expanded"
)

# YOUR KEY
API_KEY = "AIzaSyABh4pFJBdx083dBEzVRwjU0vLrccV9zZc"

# --- 3. PROFESSIONAL STYLING (CSS) ---

# This function creates the visual style of the app.
# It changes the colors and fonts based on whether you selected Light Mode or Dark Mode.
def inject_custom_css(theme_mode):
    # Base CSS defines the layout, centering the titles, and styling the boxes.
    base_css = """
<style>
    .title-container { text-align: center; padding-bottom: 30px; }
    h1 { font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; font-weight: 700; font-size: 3rem !important; margin-bottom: 0px; }
    .subtitle { font-family: 'Georgia', serif; font-size: 1.1rem; font-style: italic; opacity: 0.8; margin-top: -5px; }
    .result-card { border: 1px solid rgba(128, 128, 128, 0.2); border-radius: 8px; padding: 20px; margin-bottom: 15px; transition: 0.2s; }
    .result-card:hover { border-color: rgba(128, 128, 128, 0.5); box-shadow: 0 2px 8px rgba(0,0,0,0.05); }
    footer {visibility: hidden;}
    /* Highlight the disclaimer box to make sure users see it */
    .disclaimer-box { background-color: #fff3cd; color: #856404; padding: 10px; border-radius: 5px; border: 1px solid #ffeeba; font-size: 0.9rem; margin-top: 10px; }
</style>
"""
    # This part switches the actual colors based on your selection.
    theme_css = ""
    if theme_mode == "Light Mode":
        theme_css = """<style>[data-testid="stAppViewContainer"] { background-color: #ffffff; color: #000000; } [data-testid="stSidebar"] { background-color: #f8f9fa; border-right: 1px solid #e0e0e0; } .result-card { background-color: #ffffff; } .stMarkdown, p, h1, h2, h3 { color: #000000 !important; }</style>"""
    elif theme_mode == "Dark Mode":
        theme_css = """<style>[data-testid="stAppViewContainer"] { background-color: #0e1117; color: #ffffff; } [data-testid="stSidebar"] { background-color: #262730; border-right: 1px solid #414245; } .result-card { background-color: #1a1c24; } .stMarkdown, p, h1, h2, h3 { color: #ffffff !important; }</style>"""
    
    # Apply the styles to the app
    st.markdown(base_css + theme_css, unsafe_allow_html=True)

# --- 4. BACKEND LOGIC ---

# This function asks Google which AI brain is available for your key.
# It prevents errors by finding a working model instead of guessing.
def get_working_model():
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={API_KEY}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            # Loop through available models to find one that can generate text
            for model in data.get('models', []):
                if "generateContent" in model.get('supportedGenerationMethods', []):
                    # We prefer the 'flash' model because it is faster
                    if "flash" in model.get('name'): return model.get('name').replace("models/", "")
            return "gemini-1.5-flash"
    except:
        return "gemini-1.5-flash"

# This is the main brain of the app.
# It takes the text from the PDF, your instructions for tone and length,
# and sends it to Google to get the summary.
def analyze_paper_direct(text, tone, length):
    model_name = get_working_model()
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={API_KEY}"
    headers = {'Content-Type': 'application/json'}
    
    # This is the instruction we send to the AI
    prompt = f"""
    You are a professional research analyst. Summarize this paper.
    Configuration: Tone={tone}, Length={length}.
    
    Required Output (Markdown):
    1. **Executive Summary**
    2. **Key Findings** (Bullet points)
    3. **Methodology & Mechanics** (Explain the 'How')
    4. **Critical Analysis** (Strengths & Weaknesses)
    
    Paper Text:
    {text[:40000]}
    """
    
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    try:
        # Send the request to Google
        response = requests.post(url, headers=headers, json=payload)
        if response.status_code == 200:
            return response.json()['candidates'][0]['content']['parts'][0]['text']
        else:
            return f"Error: {response.text}"
    except Exception as e:
        return f"Connection Error: {str(e)}"

# This function opens the uploaded PDF file and pulls out all the text so the AI can read it.
# It tries multiple tools (pdfplumber, pypdf) to make sure it works.
def read_pdf(file_bytes):
    bio = io.BytesIO(file_bytes)
    texts = []
    try:
        import pdfplumber
        with pdfplumber.open(bio) as pdf:
            texts = [p.extract_text() or "" for p in pdf.pages]
    except ImportError:
        try:
            from pypdf import PdfReader
            reader = PdfReader(bio)
            texts = [p.extract_text() or "" for p in reader.pages]
        except: return ""
    return "\n".join(texts)

# --- 5. SIDEBAR ---
with st.sidebar:
    st.markdown("### Configuration")
    
    # Dropdown to choose the visual theme
    theme_choice = st.selectbox("Visual Theme", ["System Default", "Light Mode", "Dark Mode"], index=0)
    st.divider()
    
    # Dropdowns to control how the analysis is written
    st.markdown("### Analysis Parameters")
    tone = st.selectbox("Tone", ["Professional", "Academic", "Simple / ELi5"], index=0)
    length = st.selectbox("Detail Level", ["Executive Brief (Short)", "Standard Report", "Comprehensive Deep-Dive"], index=1)
    
    st.divider()
    
    # This box warns the user that the process might take a moment.
    st.markdown("""
    <div class="disclaimer-box">
        ⏳ <b>Please Wait</b><br>
        Deep analysis uses advanced AI and may take <b>30-60 seconds</b> depending on paper length.<br>
        <i>Do not refresh the page while processing.</i>
    </div>
    """, unsafe_allow_html=True)
    
    st.caption("v3.2 Stable Edition")

# Apply the chosen theme settings
inject_custom_css(theme_choice)

# --- 6. MAIN UI ---
st.markdown(f"""<div class="title-container"><h1>AVANTI</h1><div class="subtitle">The Professional Intelligence Engine</div></div>""", unsafe_allow_html=True)

# Create the two tabs for Searching and Analyzing
tab_search, tab_analyze = st.tabs(["Global Search", "Document Analysis"])

# --- SEARCH TAB ---
with tab_search:
    col1, col2 = st.columns([4, 1])
    with col1:
        query = st.text_input("Search Database", placeholder="Enter keywords, title, or DOI...")
    with col2:
        st.write("") 
        st.write("") 
        search_btn = st.button("Search", type="primary", use_container_width=True)

    if search_btn and query:
        with st.spinner(f"Querying ArXiv for '{query}'..."):
            try:
                # Ask the ArXiv database for papers matching the query
                client = requests.get(f'http://export.arxiv.org/api/query?search_query=all:{query}&start=0&max_results=5')
                feed = feedparser.parse(client.content)
                if not feed.entries:
                    st.warning("No results found.")
                else:
                    # Display each result in a nice box
                    for entry in feed.entries:
                        st.markdown(f"""
                        <div class="result-card">
                            <h4><a href="{entry.link}" target="_blank" style="text-decoration:none; color:inherit;">{entry.title}</a></h4>
                            <p style="font-size: 0.9em; opacity: 0.8;"><b>Published:</b> {entry.published[:10]} | <b>Authors:</b> {entry.authors[0].name if entry.authors else 'Unknown'}</p>
                            <p>{entry.summary[:300]}...</p>
                        </div>
                        """, unsafe_allow_html=True)
            except:
                st.error("Search failed. Please try again.")

# --- ANALYSIS TAB ---
with tab_analyze:
    uploaded_file = st.file_uploader("Upload Research Paper (PDF)", type=["pdf"])
    
    if uploaded_file:
        # We read the file immediately to check if it is valid
        raw_text = read_pdf(uploaded_file.read())
        word_count = len(raw_text.split())
        
        if word_count < 50:
            st.error("Error: Document appears empty or is an image-only PDF.")
        else:
            # Show a success message if the file is good
            st.success(f"Document Ready ({word_count:,} words)")
            
            # This button triggers the main AI analysis
            if st.button("Generate Analysis Report", type="primary", use_container_width=True):
                with st.spinner("processing logic... please wait..."):
                    summary = analyze_paper_direct(raw_text, tone, length)
                    st.divider()
                    st.markdown(summary)
                    # Let the user save the result
                    st.download_button("Download Report", summary, "Avanti_Report.md")
