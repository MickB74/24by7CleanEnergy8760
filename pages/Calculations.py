import streamlit as st
import os

# Page Config
st.set_page_config(
    page_title="Calculations - Eighty760",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize dark mode in session state if not present (should be there from main app)
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False

# Inject Custom CSS based on theme (Duplicated from app.py for consistency)
if st.session_state.dark_mode:
    # Dark Mode CSS
    st.markdown("""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=IBM+Plex+Mono:wght@700&display=swap');

            /* Dark Mode - Override everything */
            .stApp {
                background-color: #0E1117 !important;
            }
            .main .block-container {
                background-color: #0E1117 !important;
            }
            section[data-testid="stSidebar"] {
                background-color: #262730 !important;
            }
            section[data-testid="stSidebar"] > div {
                background-color: #262730 !important;
            }
            
            /* Text Colors */
            .stApp, .stApp p, .stApp label, .stApp span, .stApp div {
                color: #FAFAFA !important;
            }
            h1, h2, h3, h4, h5, h6 {
                color: #FAFAFA !important;
                font-weight: 700;
            }
            
            /* Global Typography */
            html, body, [class*="css"] {
                font-family: 'Inter', sans-serif;
            }
            
            /* Layout Adjustments */
            .block-container {
                padding-top: 2rem;
                padding-bottom: 2rem;
                max-width: 1440px;
            }

            /* Markdown Styling */
            .stMarkdown p {
                font-size: 1.05rem;
                line-height: 1.6;
            }
            .stMarkdown h1 {
                font-size: 2.5rem;
                margin-bottom: 1.5rem;
                color: #00D9FF !important;
            }
            .stMarkdown h2 {
                font-size: 1.8rem;
                margin-top: 2rem;
                margin-bottom: 1rem;
                border-bottom: 1px solid rgba(250, 250, 250, 0.2);
                padding-bottom: 0.5rem;
            }
            .stMarkdown h3 {
                font-size: 1.4rem;
                margin-top: 1.5rem;
                color: #B0B0B0 !important;
            }
            
            /* Code Blocks */
            code {
                color: #00D9FF !important;
                background-color: rgba(255, 255, 255, 0.1) !important;
                padding: 0.2rem 0.4rem;
                border-radius: 4px;
            }
            
            /* Blockquotes */
            blockquote {
                border-left: 3px solid #00D9FF !important;
                background-color: rgba(0, 217, 255, 0.05) !important;
                padding: 1rem !important;
                color: #FAFAFA !important;
            }
        </style>
    """, unsafe_allow_html=True)
else:
    # Light Mode CSS
    st.markdown("""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=IBM+Plex+Mono:wght@700&display=swap');

            html, body, [class*="css"] {
                font-family: 'Inter', sans-serif;
            }
            h1, h2, h3 {
                font-weight: 700;
            }
            
            .block-container {
                padding-top: 2rem;
                padding-bottom: 2rem;
                max-width: 1440px;
            }
            
            /* Markdown Styling */
            .stMarkdown h1 {
                color: #285477;
            }
            .stMarkdown h2 {
                color: #285477;
                border-bottom: 1px solid #E0E0E0;
                padding-bottom: 0.5rem;
            }
        </style>
    """, unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.toggle("Dark Mode", key="dark_mode")
    st.markdown("---")
    if st.button("← Back to Simulator", use_container_width=True):
        st.switch_page("app.py")

# Main Content
try:
    # Read the CALCULATIONS.md file
    # Assuming the file is in the root directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(current_dir)
    file_path = os.path.join(root_dir, "CALCULATIONS.md")
    
    with open(file_path, "r") as f:
        content = f.read()
    
    # Render the content
    st.markdown(content, unsafe_allow_html=True)
    
except FileNotFoundError:
    st.error("⚠️ CALCULATIONS.md file not found in the root directory.")
except Exception as e:
    st.error(f"❌ Error reading calculations file: {e}")
