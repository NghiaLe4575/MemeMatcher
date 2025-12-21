"""CSS styling for MemeMatcher application"""
import streamlit as st


def inject_styles():
    """Inject custom CSS styles into the Streamlit app"""
    st.markdown("""
        <style>
            /* Global font and base light background */
            html, body, [class*="css"] {
                font-family: 'Inter', 'Segoe UI', sans-serif;
                background: linear-gradient(135deg, #0A2540 0%, #1A3A5C 100%) !important;
                color: #2C3E50;
            }
            /* Ensure Streamlit app container uses light background */
            [data-testid="stAppViewContainer"] {
                background: linear-gradient(135deg, #0A2540 0%, #1A3A5C 100%) !important;
            }

            /* Main container */
            .main .block-container {
                padding-top: 2rem;
                max-width: 1400px;
            }

            /* Smooth rounded corners */
            .stTextInput input, .stSelectbox div, .stButton button, .stSlider {
                border-radius: 12px !important;
                transition: all 0.3s ease;
            }

            /* Modern button styling - Colorful gradients */
            .stButton button {
                background: linear-gradient(135deg, #0046FF 0%, #00C9FF 100%);
                color: #FFFFFF;
                font-weight: 700;
                border: none;
                box-shadow: 0 4px 12px rgba(0, 70, 255, 0.3);
                padding: 0.75rem 2rem;
                font-size: 1rem;
                letter-spacing: 0.5px;
            }
            .stButton button:hover {
                transform: translateY(-2px);
                box-shadow: 0 8px 20px rgba(0, 70, 255, 0.4);
                background: linear-gradient(135deg, #0046FF 0%, #00C9FF 100%);
            }
            /* Header styling - White on dark blue */
            h1 { 
                color: #FFFFFF; 
                font-weight: 800; 
                letter-spacing: -1px;
                text-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
            }
            h2 { color: #1A3A5C; font-weight: 700; }
            h3 { color: #2C3E50; font-weight: 700; }

            /* Card styling for results - Clean white with colorful accents */
            .meme-card {
                background: white;
                border-radius: 16px;
                padding: 1.5rem;
                box-shadow: 0 4px 16px rgba(0, 0, 0, 0.08);
                transition: all 0.3s ease;
                border: 2px solid #E8EEF5;
            }
            .meme-card:hover {
                transform: translateY(-6px);
                box-shadow: 0 12px 28px rgba(0, 70, 255, 0.15);
                border-color: #0046FF;
            }

            /* Badge styling - Colorful badges */
            .badge {
                display: inline-block;
                padding: 0.4rem 1rem;
                border-radius: 20px;
                font-size: 0.85rem;
                font-weight: 700;
                margin: 0.25rem;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }
            .badge-usage { 
                background: linear-gradient(135deg, #00C9FF 0%, #0046FF 100%);
                color: #FFFFFF;
                box-shadow: 0 2px 8px rgba(0, 70, 255, 0.3);
            }
            .badge-visual { 
                background: linear-gradient(135deg, #00C9FF 0%, #00C9FF 100%);
                color: #FFFFFF;
                box-shadow: 0 2px 8px rgba(255, 144, 19, 0.3);
            }

            /* Context/usage text styling */
            .usage-context {
                background: linear-gradient(to right, #FFF5E6 0%, #FFFFFF 100%);
                border-left: 5px solid #00C9FF;
                padding: 1rem 1.25rem;
                border-radius: 12px;
                font-size: 0.95rem;
                color: #2C3E50;
                margin: 0.75rem 0;
                line-height: 1.6;
                box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
            }

            /* Search section styling - minimal (avoid empty white banner) */
            .search-section {
                background: transparent;
                border-radius: 0;
                padding: 0;
                box-shadow: none;
                margin-bottom: 1rem;
            }

            /* Input styling - prominent yet refined */
            .stTextInput input {
                border: 1.5px solid #D5DEED !important;
                font-size: 1.25rem !important;
                padding: 1rem 1.25rem !important;
                color: #1A3A5C !important;
                background-color: #FFFFFF !important;
                font-weight: 600 !important;
                box-shadow: 0 2px 6px rgba(0, 70, 255, 0.08) !important;
                outline: none !important;
            }
            .stTextInput input:hover {
                border-color: #0046FF !important;
            }
            .stTextInput input:focus,
            .stTextInput input:focus-visible,
            .stTextInput input:focus-within,
            .stTextInput input:active {
                border-color: #0046FF !important;
                box-shadow: 0 0 0 2px rgba(0, 70, 255, 0.2) !important;
                outline: none !important;
                outline-width: 0 !important;
                outline-color: transparent !important;
            }
            /* Remove any Streamlit default focus rings */
            .stTextInput > div > div {
                border: none !important;
                box-shadow: none !important;
            }
            .stTextInput > div > div:focus-within {
                border: none !important;
                box-shadow: none !important;
            }
            .stTextInput input::placeholder {
                color: #8CA3C0 !important;
                opacity: 1 !important;
            }

            /* Radio buttons styling - Modern pills */
            .stRadio > div {
                gap: 12px;
            }
            .stRadio > div > label {
                background: rgba(255, 255, 255, 0.1);
                border: 2px solid rgba(255, 255, 255, 0.3);
                border-radius: 25px;
                padding: 0.6rem 1.5rem;
                color: #FFFFFF;
                font-weight: 600;
                transition: all 0.3s ease;
                cursor: pointer;
            }
            .stRadio > div > label:hover {
                background: rgba(255, 255, 255, 0.2);
                border-color: #00C9FF;
                transform: translateY(-2px);
            }
            /* Selected radio button - highlighted with solid background */
            .stRadio > div > label:has(input:checked) {
                background: linear-gradient(135deg, #0046FF 0%, #00C9FF 100%) !important;
                border-color: #FFFFFF !important;
                box-shadow: 0 4px 12px rgba(0, 201, 255, 0.4);
                transform: scale(1.05);
                font-weight: 700;
            }
            .stRadio > div > label[data-baseweb="radio"] > div:first-child {
                display: none;
            }

            /* Metrics styling */
            [data-testid="stMetricValue"] {
                color: #FFFFFF;
                font-size: 2rem;
                font-weight: 800;
            }
            [data-testid="stMetricLabel"] {
                color: rgba(255, 255, 255, 0.8);
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 1px;
                font-size: 0.85rem;
            }

            /* Slider styling */
            .stSlider {
                padding: 1rem 0;
            }
            .stSlider > div > div > div {
                color: #FFFFFF !important;
            }

            /* Expander styling */
            .streamlit-expanderHeader {
                background: rgba(255, 255, 255, 0.1);
                border: 2px solid rgba(255, 255, 255, 0.2);
                border-radius: 12px;
                font-weight: 600;
                color: #FFFFFF;
                padding: 0.75rem 1rem;
            }
            .streamlit-expanderHeader:hover {
                background: rgba(255, 255, 255, 0.15);
                border-color: #00C9FF;
            }

            /* Selectbox styling (history) - compact */
            .stSelectbox > div > div {
                background-color: white;
                border: 1.5px solid #D5DEED;
                border-radius: 10px;
                color: #2C3E50;
                min-height: 42px;
            }
            .stSelectbox > div > div:hover {
                border-color: #0046FF;
            }

            /* File uploader styling */
            [data-testid="stFileUploader"] {
                background: white;
                border: 3px dashed #0046FF;
                border-radius: 16px;
                padding: 2rem;
            }
            [data-testid="stFileUploader"]:hover {
                border-color: #00C9FF;
                background: #F8FBFF;
            }

            /* Progress bar styling */
            .stProgress > div > div > div {
                background: linear-gradient(90deg, #0046FF 0%, #00C9FF 100%);
            }

            /* Caption styling */
            .css-1544g2n, .css-nahz7x {
                color: #64748B;
                font-size: 0.875rem;
            }

            /* Hide Streamlit branding */
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            
            /* History chip buttons - subtle secondary style */
            button[kind="secondary"] {
                background: rgba(255, 255, 255, 0.15) !important;
                border: 1px solid rgba(255, 255, 255, 0.3) !important;
                color: #FFFFFF !important;
                font-size: 0.85rem !important;
                padding: 0.4rem 0.8rem !important;
                border-radius: 20px !important;
            }
            button[kind="secondary"]:hover {
                background: rgba(0, 70, 255, 0.3) !important;
                border-color: #0046FF !important;
            }
            
            /* Column gap adjustment */
            [data-testid="column"] {
                padding: 0.5rem;
            }

            /* Divider styling */
            hr {
                margin: 2rem 0;
                border: none;
                height: 2px;
                background: linear-gradient(90deg, transparent 0%, #E8EEF5 50%, transparent 100%);
            }
        </style>
    """, unsafe_allow_html=True)
