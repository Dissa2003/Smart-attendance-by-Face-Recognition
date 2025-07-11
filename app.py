import streamlit as st
import pandas as pd
import time
from datetime import datetime
import os
from streamlit_autorefresh import st_autorefresh

# Auto-refresh every 2 seconds
count = st_autorefresh(interval=2000, limit=None, key="auto-refresh")

# Use current date in correct format (YYYY-MM-DD)
date = datetime.now().strftime("%Y-%m-%d")
filename = f"Attendence/Attendance_{date}.csv"

st.title("📋 Face Recognition Attendance")
st.write(f"📅 Date: {date}")
st.write(f"🔄 Auto-refresh count: {count}")

# Check if the file exists
if os.path.exists(filename):
    df = pd.read_csv(filename)
    st.success("✅ Attendance file loaded")
    st.dataframe(df.style.highlight_max(axis=0))
else:
    st.warning("⚠️ Attendance file not found yet. Please mark attendance using the camera system.")
