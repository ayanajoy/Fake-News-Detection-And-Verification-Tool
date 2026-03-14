import streamlit as st
import requests
import pandas as pd
import time
import matplotlib.pyplot as plt

# ---------------- PAGE CONFIG ---------------- #

st.set_page_config(
    page_title="Fake News Detection and Verification Tool",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- PROFESSIONAL CSS ---------------- #

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.main {
    background-color:#f4f6fb;
}

/* Card */
.card{
background:white;
padding:1.5rem;
border-radius:12px;
border:1px solid #e6e8eb;
box-shadow:0px 4px 12px rgba(0,0,0,0.05);
margin-bottom:20px;
}

/* Sidebar */
[data-testid="stSidebar"]{
background-color:#0f172a;
}

[data-testid="stSidebar"] p{
color:#94a3b8;
}

/* Button */
.stButton>button{
width:100%;
border-radius:8px;
height:3em;
background:#2563eb;
color:white;
font-weight:600;
border:none;
}

.stButton>button:hover{
background:#1d4ed8;
}

/* Suspicious phrases */
.suspicious{
background:#fee2e2;
border-left:4px solid #dc2626;
padding:10px;
border-radius:6px;
color:#7f1d1d;
font-weight:600;
}

/* Trusted */
.trusted{
background:#dcfce7;
border-left:4px solid #16a34a;
padding:10px;
border-radius:6px;
color:#065f46;
font-weight:600;
}

/* Untrusted */
.untrusted{
background:#fee2e2;
border-left:4px solid #dc2626;
padding:10px;
border-radius:6px;
color:#7f1d1d;
font-weight:600;
}

</style>
""", unsafe_allow_html=True)

# ---------------- BACKEND API ---------------- #

API = "http://127.0.0.1:5000/analyze"

# ---------------- SESSION DATA ---------------- #

if 'history' not in st.session_state:
    st.session_state.history = []

if 'stats' not in st.session_state:
    st.session_state.stats = {"fake":0,"real":0,"claims":0}

# ---------------- SIDEBAR ---------------- #

with st.sidebar:

    st.title("🛡️ Admin Dashboard")

    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["🚀 Analyze News","📊 System Overview","🕒 Analysis History"]
    )

    st.markdown("---")

    st.caption("System Status: Operational")
    st.caption("API Endpoint: 127.0.0.1:5000")

    st.markdown("---")

    st.markdown("### Model Info")
    st.caption("Model: Fake News Classifier")
    st.caption("Explainability: Rule-based")
    st.caption("Deployment: Docker Container")

# ---------------- ANALYZE PAGE ---------------- #

def show_analyze():

    st.title("🚀 Fake News Analyzer")

    st.write("Analyze news articles using AI detection and explainability.")

    col1,col2 = st.columns([1,1])

    with col1:

        text = st.text_area(
            "News Article",
            height=250,
            placeholder="Paste the news article here..."
        )

        source = st.text_input(
            "News Source",
            placeholder="example.com"
        )

        analyze_btn = st.button("Run Analysis")

    with col2:

        if analyze_btn and text:

            with st.spinner("Running AI analysis..."):

                start_time = time.time()

                try:

                    response = requests.post(API,json={"text":text,"source":source})
                    res = response.json()

                    proc_time = round(time.time()-start_time,2)

                    label = res["prediction"]["prediction"]
                    confidence = res["prediction"]["confidence"]

                    explanation = res["explanation"]
                    suspicious = res["suspicious_phrases"]
                    trusted = res["trusted_source"]

                    # ----- UPDATE STATS -----

                    if label.upper()=="FAKE":
                        st.session_state.stats["fake"]+=1
                    else:
                        st.session_state.stats["real"]+=1

                    st.session_state.stats["claims"]+=len(suspicious)

                    # ----- SAVE HISTORY -----

                    st.session_state.history.append({
                        "Time":time.strftime("%H:%M:%S"),
                        "Source":source,
                        "Result":label,
                        "Confidence":confidence,
                        "Text":text[:80]+"..."
                    })

                    # ----- RESULT -----

                    st.markdown("### Analysis Result")

                    if label.upper()=="FAKE":
                        st.error(f"🚩 Prediction: {label}")
                    else:
                        st.success(f"✅ Prediction: {label}")

                    st.progress(confidence,text=f"Model Confidence: {int(confidence*100)}%")

                    st.caption(f"Processing Time: {proc_time} seconds")

                    # ----- AI EXPLANATION -----

                    st.markdown("### AI Explanation")

                    if explanation:
                        for e in explanation:
                            st.info(e)

                    # ----- SUSPICIOUS CLAIMS -----

                    if suspicious:

                        st.markdown("### Suspicious Claims Detected")

                        for phrase in suspicious:
                            st.markdown(
                                f"<div class='suspicious'>{phrase}</div>",
                                unsafe_allow_html=True
                            )

                    else:

                        st.success("No suspicious phrases detected")

                    # ----- SOURCE TRUST -----

                    st.markdown("### Source Credibility")

                    if trusted:
                         st.markdown(
                            "<div class='trusted'>✔ Trusted News Source</div>",
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            "<div class='untrusted'>⚠ Unverified or Suspicious Source</div>",
                            unsafe_allow_html=True
                        )

                except Exception as e:

                    st.error("Backend API not running")
                    st.write(e)

        else:

            st.info("Enter article text and click **Run Analysis**")

# ---------------- OVERVIEW PAGE ---------------- #

def show_overview():

    st.title("📊 System Analytics")

    m1,m2,m3,m4 = st.columns(4)

    total = st.session_state.stats["fake"] + st.session_state.stats["real"]

    m1.metric("Total Articles",total)
    m2.metric("Fake News",st.session_state.stats["fake"])
    m3.metric("Real News",st.session_state.stats["real"])
    m4.metric("Claims Flagged",st.session_state.stats["claims"])

    st.markdown("---")

    if st.session_state.history:

        df = pd.DataFrame(st.session_state.history)

        st.subheader("Fake vs Real Distribution")

        chart_data = df["Result"].value_counts()

        st.bar_chart(chart_data)

        fig, ax = plt.subplots()
        chart_data.plot.pie(autopct='%1.1f%%',ax=ax)
        ax.set_ylabel("")
        st.pyplot(fig)

    else:

        st.info("No analytics available yet")


def show_history():

    st.title("🕒 Analysis History")

    if st.session_state.history:

        df = pd.DataFrame(st.session_state.history).iloc[::-1]

        search = st.text_input("Search history")

        if search:

            df = df[
                df["Source"].str.contains(search,case=False) |
                df["Text"].str.contains(search,case=False)
            ]

        st.dataframe(df,use_container_width=True)

        csv = df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download CSV",
            csv,
            "analysis_history.csv",
            "text/csv"
        )

    else:

        st.info("No analysis performed yet")

if page=="🚀 Analyze News":
    show_analyze()

elif page=="📊 System Overview":
    show_overview()

elif page=="🕒 Analysis History":
    show_history()