import streamlit as st
import requests
import os
from typing import List, Dict, Any

# Page Configuration
st.set_page_config(
    page_title="Appliance Recommender",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Minimalist Custom Styling
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #111827;
    }
    
    .header-section {
        padding-bottom: 20px;
        margin-bottom: 24px;
        border-bottom: 1px solid #e5e7eb;
    }
    
    .header-title {
        font-size: 1.8rem;
        font-weight: 700;
        color: #111827;
        letter-spacing: -0.02em;
        margin-bottom: 4px;
    }
    
    .header-subtitle {
        font-size: 0.95rem;
        color: #6b7280;
        margin-bottom: 12px;
    }
    
    .stat-tag {
        display: inline-block;
        background: #f3f4f6;
        color: #374151;
        font-size: 0.78rem;
        font-weight: 500;
        padding: 3px 10px;
        border-radius: 4px;
        margin-right: 8px;
    }
    
    .product-title {
        font-size: 0.9rem;
        font-weight: 600;
        color: #111827;
        line-height: 1.35;
        margin-top: 10px;
        margin-bottom: 8px;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
        overflow: hidden;
        height: 2.6em;
    }
    
    .badge-match {
        font-size: 0.75rem;
        font-weight: 600;
        color: #1d4ed8;
        background: #eff6ff;
        border: 1px solid #dbeafe;
        padding: 2px 8px;
        border-radius: 4px;
        display: inline-block;
    }
    
    .badge-price {
        font-size: 1rem;
        font-weight: 600;
        color: #111827;
    }
    
    .badge-rating {
        font-size: 0.8rem;
        color: #4b5563;
        font-weight: 500;
    }
    
    .category-chip {
        font-size: 0.72rem;
        color: #6b7280;
        background: #f9fafb;
        border: 1px solid #e5e7eb;
        padding: 2px 6px;
        border-radius: 3px;
    }
    
    .meta-text {
        font-size: 0.75rem;
        color: #9ca3af;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Backend URL Configuration
DEFAULT_API_URL = os.environ.get("BACKEND_URL", "http://127.0.0.1:8000")

# Sidebar
with st.sidebar:
    st.markdown("### Settings")
    api_url = st.text_input("API Endpoint", value=DEFAULT_API_URL).rstrip("/")
    
    # Live backend health probe
    backend_online = False
    system_stats = {}
    try:
        r = requests.get(f"{api_url}/health", timeout=2)
        if r.status_code == 200:
            backend_online = True
            system_stats = r.json()
            st.caption("Backend: Connected")
        else:
            st.caption(f"Backend: Status {r.status_code}")
    except Exception:
        st.caption("Backend: Offline")
    
    st.divider()
    st.markdown("### Filters")
    top_k = st.slider("Results count", min_value=3, max_value=24, value=9, step=3)
    min_rating = st.slider("Minimum rating", min_value=0.0, max_value=5.0, value=3.5, step=0.5)
    
    if backend_online and "total_products" in system_stats:
        st.divider()
        st.metric("Catalog size", f"{system_stats['total_products']:,}")
        if "stats" in system_stats and "average_rating" in system_stats["stats"]:
            st.metric("Average rating", f"{system_stats['stats']['average_rating']} / 5.0")

# Minimal Header
st.markdown(
    """
    <div class="header-section">
        <div class="header-title">Appliance Recommender</div>
        <div class="header-subtitle">
            Search and discovery engine for Amazon Appliances powered by TF-IDF and Nearest Neighbors.
        </div>
        <div>
            <span class="stat-tag">65,743 products</span>
            <span class="stat-tag">Content-based matching</span>
            <span class="stat-tag">Sub-second response</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Search Tabs
tab_search, tab_catalog = st.tabs([
    "Natural Language Search",
    "Catalog Seed Search",
])

# Clean Sample Prompts
sample_queries = [
    "Countertop portable ice maker machine",
    "Refrigerator water filter replacement cartridge",
    "LG washing machine drain pump hose assembly",
    "Dishwasher silverware basket cutlery holder",
    "Microwave glass turntable plate replacement",
]

# Mode 1: Natural Language Search
with tab_search:
    col_input, col_btn = st.columns([5, 1])
    with col_input:
        search_query = st.text_input(
            "Query",
            placeholder="Search by appliance description or part specifications...",
            label_visibility="collapsed",
        )
    with col_btn:
        search_clicked = st.button("Search", type="primary", use_container_width=True)

    # Minimal text pills for query suggestions
    st.caption("Suggestions:")
    sample_cols = st.columns(len(sample_queries))
    for i, sq in enumerate(sample_queries):
        if sample_cols[i].button(sq, key=f"sq_{i}", use_container_width=True):
            search_query = sq
            search_clicked = True

    if search_clicked and search_query:
        if not backend_online:
            st.error(f"Backend unavailable at {api_url}. Start the server with: python main.py --serve")
        else:
            with st.spinner("Searching..."):
                try:
                    resp = requests.get(
                        f"{api_url}/recommend/query",
                        params={"query": search_query, "top_k": top_k * 2},
                        timeout=10,
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        recs = data.get("recommendations", [])
                        filtered_recs = [r for r in recs if r.get("average_rating", 0) >= min_rating][:top_k]
                        
                        st.markdown(f"**Results for:** *\"{search_query}\"* ({len(filtered_recs)} products)")
                        
                        if not filtered_recs:
                            st.info("No products match the selected rating filter. Try adjusting the minimum rating slider.")
                        else:
                            # Minimal 3-column Grid
                            cols_per_row = 3
                            for row_idx in range(0, len(filtered_recs), cols_per_row):
                                cols = st.columns(cols_per_row)
                                for col_idx in range(cols_per_row):
                                    item_idx = row_idx + col_idx
                                    if item_idx < len(filtered_recs):
                                        prod = filtered_recs[item_idx]
                                        with cols[col_idx]:
                                            with st.container(border=True):
                                                img = prod.get("image_url") or "https://images.unsplash.com/photo-1584269600464-37b1b58a9fe7?auto=format&fit=crop&w=600&q=80"
                                                st.image(img, use_container_width=True)
                                                
                                                # Top info row
                                                b_col1, b_col2 = st.columns([1, 1])
                                                with b_col1:
                                                    match_pct = prod.get("match_percentage", "90%")
                                                    st.markdown(f"<span class='badge-match'>{match_pct} match</span>", unsafe_allow_html=True)
                                                with b_col2:
                                                    st.markdown(f"<div style='text-align: right;' class='badge-price'>{prod.get('price', '$ --')}</div>", unsafe_allow_html=True)
                                                
                                                st.markdown(f"<div class='product-title' title='{prod.get('title')}'>{prod.get('title')}</div>", unsafe_allow_html=True)
                                                
                                                # Rating and Category
                                                r_col1, r_col2 = st.columns([1, 1])
                                                with r_col1:
                                                    rating = prod.get("average_rating", 0)
                                                    st.markdown(f"<span class='badge-rating'>Rating: {rating:.1f} ({prod.get('rating_number', 0)})</span>", unsafe_allow_html=True)
                                                with r_col2:
                                                    cats = prod.get("categories", ["Appliance"])
                                                    cat_name = cats[-1] if cats else "Appliance"
                                                    st.markdown(f"<div style='text-align: right;'><span class='category-chip'>{cat_name}</span></div>", unsafe_allow_html=True)
                                                
                                                st.caption(f"ASIN: {prod.get('parent_asin')} | Brand: {prod.get('store', 'Amazon')}")
                    else:
                        st.error(f"Error ({resp.status_code}): {resp.text}")
                except Exception as e:
                    st.error(f"Request failed: {e}")

# Mode 2: Catalog Seed Search
with tab_catalog:
    col_seed, col_seed_btn = st.columns([5, 1])
    with col_seed:
        seed_input = st.text_input(
            "Product name or ASIN",
            value="Samsung Refrigerator",
            placeholder="Enter an existing catalog title or ASIN...",
            label_visibility="collapsed",
        )
    with col_seed_btn:
        seed_clicked = st.button("Find Similar", type="primary", use_container_width=True)

    if seed_clicked and seed_input:
        if not backend_online:
            st.error(f"Backend unavailable at {api_url}.")
        else:
            with st.spinner("Finding similar items..."):
                try:
                    resp = requests.get(
                        f"{api_url}/recommend",
                        params={"product_title": seed_input, "top_k": top_k * 2},
                        timeout=10,
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        recs = data.get("recommendations", [])
                        filtered_recs = [r for r in recs if r.get("average_rating", 0) >= min_rating][:top_k]
                        
                        st.markdown(f"**Similar to:** *\"{seed_input}\"* ({len(filtered_recs)} products)")
                        
                        cols_per_row = 3
                        for row_idx in range(0, len(filtered_recs), cols_per_row):
                            cols = st.columns(cols_per_row)
                            for col_idx in range(cols_per_row):
                                item_idx = row_idx + col_idx
                                if item_idx < len(filtered_recs):
                                    prod = filtered_recs[item_idx]
                                    with cols[col_idx]:
                                        with st.container(border=True):
                                            img = prod.get("image_url") or "https://images.unsplash.com/photo-1584269600464-37b1b58a9fe7?auto=format&fit=crop&w=600&q=80"
                                            st.image(img, use_container_width=True)
                                            
                                            b_col1, b_col2 = st.columns([1, 1])
                                            with b_col1:
                                                match_pct = prod.get("match_percentage", "88%")
                                                st.markdown(f"<span class='badge-match'>{match_pct} match</span>", unsafe_allow_html=True)
                                            with b_col2:
                                                st.markdown(f"<div style='text-align: right;' class='badge-price'>{prod.get('price', '$ --')}</div>", unsafe_allow_html=True)
                                            
                                            st.markdown(f"<div class='product-title' title='{prod.get('title')}'>{prod.get('title')}</div>", unsafe_allow_html=True)
                                            
                                            r_col1, r_col2 = st.columns([1, 1])
                                            with r_col1:
                                                rating = prod.get("average_rating", 0)
                                                st.markdown(f"<span class='badge-rating'>Rating: {rating:.1f} ({prod.get('rating_number', 0)})</span>", unsafe_allow_html=True)
                                            with r_col2:
                                                cats = prod.get("categories", ["Appliance"])
                                                cat_name = cats[-1] if cats else "Appliance"
                                                st.markdown(f"<div style='text-align: right;'><span class='category-chip'>{cat_name}</span></div>", unsafe_allow_html=True)
                                            
                                            st.caption(f"ASIN: {prod.get('parent_asin')} | Brand: {prod.get('store', 'Amazon')}")
                    else:
                        st.error(f"Error ({resp.status_code}): {resp.text}")
                except Exception as e:
                    st.error(f"Request failed: {e}")