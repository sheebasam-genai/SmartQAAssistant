import streamlit as st

def show_docs():

    st.title("📘 SMART QA Assistant")
    st.subheader("AI-Powered QA Intelligence Platform")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Overview",
        "Features",
        "Query Demo",
        "Architecture",
        "Future Roadmap"
    ])

    with tab1:
        st.header("🚀 Overview")
        st.write("""
        SMART QA Assistant helps QA teams:
        - Query defects using natural language
        - Get AI insights (root cause, resolution)
        - Detect patterns and trends
        - Create defects directly
        """)
        st.success("👉 No filters. No SQL. Just ask in English.")

    with tab2:
        st.header("🧩 Features")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 🔹 Core")
            st.markdown("""
            - Natural language queries  
            - Defect filtering  
            - History tracking  
            - Token dashboard  
            """)

        with col2:
            st.markdown("### 🤖 AI")
            st.markdown("""
            - Title generation  
            - Root cause analysis  
            - Resolution suggestion  
            - Pattern detection  
            """)

    with tab3:
        st.header("🧠 Query Examples")

        query = st.selectbox("Try Sample Queries", [
            "Show all P1 defects",
            "List defects assigned to SUBHRA",
            "Find defects with TCode ME21N",
            "Show defects from release R2",
            "Show closed defects",
            "Summarize defect COTEST-0002",
            "What are the most common defects?"
        ])

        st.code(query)

        if st.button("Run Query (Demo)"):
            st.info("👉 Connect this to your actual query engine")

    with tab4:
        st.header("🏗️ Architecture")
        st.code("""
User
 ↓
Streamlit UI
 ↓
Query Classifier
 ↓
Rule Engine | AI Engine | Analytics Engine
 ↓
Pandas Processing
 ↓
CSV (KEDB)
 ↓
ChromaDB
 ↓
LLM
        """)

    with tab5:
        st.header("🔮 Future Roadmap")
        st.markdown("""
        - JIRA integration  
        - Zephyr integration  
        - Predictive defect analysis  
        - CI/CD integration  
        - Advanced dashboards  
        """)

# 🔥 THIS IS THE FIX
if __name__ == "__main__":
    show_docs()