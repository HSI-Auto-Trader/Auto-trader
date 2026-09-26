import streamlit as st

if st.sidebar.button("🔄 Fetch Live Market Data"):
    st.cache_data.clear()  # Clear Streamlit cache
    df_live = fetch_live_h1_candles()
    st.success("Fetched latest H1 candle!")