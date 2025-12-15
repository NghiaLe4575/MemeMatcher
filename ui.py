import streamlit as st
import pandas as pd
import time

# 1. PAGE CONFIG: Wide mode, serious title
st.set_page_config(
    page_title="EVSD | Enterprise Visual Semiotics Database",
    page_icon="🗄️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. CUSTOM CSS: Remove roundness, force fonts, make it look "legacy"
st.markdown("""
    <style>
        /* Force font to generic sans-serif or monospace */
        html, body, [class*="css"] {
            font-family: 'Arial', sans-serif;
        }
        
        /* Remove rounded corners from buttons and inputs */
        .stTextInput input, .stSelectbox div, .stButton button {
            border-radius: 0px !important;
            border: 1px solid #ccc;
        }
        
        /* Make buttons look utilitarian */
        .stButton button {
            background-color: #f0f0f0;
            color: black;
            font-weight: bold;
            border: 1px solid #999;
            box-shadow: none;
        }
        .stButton button:hover {
            background-color: #e0e0e0;
            border-color: #666;
        }

        /* Sidebar styling */
        section[data-testid="stSidebar"] {
            background-color: #160414;
            border-right: 1px solid #ccc;
        }
        
        /* Header styling */
        h1, h2, h3 {
            color: #2c3e50;
            font-weight: 600;
            letter-spacing: -0.5px;
        }
        
        /* Hide Streamlit branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# 3. SIDEBAR: System Status & Filters
with st.sidebar:
    st.markdown("### 🔒 SYSTEM ACCESS: GRANTED")
    st.code("ID: ADMIN_ROOT\nSESSION: #884-XJ-99\nSECURE CONNECTION: TRUE")
    
    st.markdown("---")
    st.markdown("**QUERY PARAMETERS**")
    
    query_mode = st.selectbox("Search Algorithm", ["Semantic Vector Match (Cosine)", "Keyword Boolean", "OCR Text Extraction"])
    sensitivity = st.slider("Irony Threshold", 0.0, 1.0, 0.75)
    file_type = st.multiselect("Asset Format", ["JPG", "PNG", "GIF", "WEBP"], default=["JPG", "PNG"])
    
    st.markdown("---")
    st.warning("⚠️ PROPRIETARY DATA. DO NOT DISTRIBUTE.")

# 4. MAIN DASHBOARD
st.title("Enterprise Visual Semiotics Database (EVSD)")
st.caption("v2.4.1 | Build 9942 | Local_Host_Node")

# Metrics Row
c1, c2, c3, c4 = st.columns(4)
c1.metric("Indexed Artifacts", "14,502", "+12")
c2.metric("Query Latency", "0.04s", "-0.01s")
c3.metric("Storage Utilization", "42.8 GB", "Stable")
c4.metric("Humor Efficiency", "14.2%", "-2.1%")

st.markdown("---")

# 5. SEARCH INTERFACE
with st.form("search_form"):
    c_search, c_btn = st.columns([5, 1])
    with c_search:
        query = st.text_input("Enter Query Parameters", placeholder="e.g., 'feline', 'existential_dread', 'success_kid'")
    with c_btn:
        st.write("") # Spacer
        st.write("") 
        submitted = st.form_submit_button("EXECUTE QUERY")

# 6. RESULTS DISPLAY
if submitted:
    with st.spinner("Processing Semantic Vectors..."):
        time.sleep(0.8) # Fake loading time for "realism"
    
    st.markdown("### Search Results (Confidence > 80%)")
    
    # Mock Result 1
    with st.container():
        col_img, col_data = st.columns([1, 2])
        with col_img:
            # Placeholder for a meme
            st.image("https://i.imgflip.com/2/1bij.jpg", caption="Asset_ID: 004921.jpg", width=300)
        with col_data:
            st.markdown("#### 📄 Asset Analysis Report")
            st.table(pd.DataFrame({
                "Attribute": ["Classification", "Detected Text", "Emotive Vector", "Origin Date"],
                "Value": ["CLASSIC_MACRO", "ONE DOES NOT SIMPLY WALK INTO MORDOR", "WARNING/STOIC", "2004-02-12"]
            }))
            st.info("ℹ️ NOTE: Asset exhibits high virality in Q3 2010. Cultural relevance is currently depreciating.")

    st.markdown("---")

    # Mock Result 2
    with st.container():
        col_img, col_data = st.columns([1, 2])
        with col_img:
            st.image("https://i.imgflip.com/30b1gx.jpg", caption="Asset_ID: 994211.jpg", width=300)
        with col_data:
            st.markdown("#### 📄 Asset Analysis Report")
            st.table(pd.DataFrame({
                "Attribute": ["Classification", "Detected Text", "Emotive Vector", "Origin Date"],
                "Value": ["DRAKE_FORMAT", "NAIVE VS PREFERRED", "APPROVAL/REJECTION", "2015-10-20"]
            }))
            st.error("⚠️ ALERT: High frequency of reuse detected. Saturation imminent.")

else:
    st.info("System Ready. Awaiting Input.")