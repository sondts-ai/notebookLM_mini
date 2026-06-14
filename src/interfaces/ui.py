import streamlit as st
import httpx

from src.config import settings
from src.interfaces.styles import GLOBAL_CSS

_API = settings.api_url

def _api(method,path,**kwargs):
    return httpx.request(
        method,
        f"{_API}{path}",
        timeout=120,
        **kwargs
    )

def _sidebar():
    st.sidebar.title("RAG Learning System")

    uploaded = st.sidebar.file_uploader(
        "Upload PDF",
        type=["pdf"],
    )

    if uploaded:
        if st.sidebar.button("Upload"):
            files = {
                "file": (
                    uploaded.name,
                    uploaded.getvalue(),
                    "application/pdf",
                )
            }

            res = _api("POST", "/upload", files=files)

            if res.status_code == 200:
                st.sidebar.success("Upload thành công")
            else:
                st.sidebar.error("Upload thất bại")
                st.sidebar.write(res.text)

    return [], None

def _tab_chat(filenames,page):
    question=st.text_input(
        "Question"
    )
    if st.button("ask"):
        payload={
            "question":question
        }
        res=_api("POST","/ask",json=payload)

        if res.status_code==200:
            st.sidebar.success("up load thanh cong")

    return [],None

def _tab_chat(filenames, page):

    question = st.text_input(
        "Question"
    )

    if st.button("Ask"):
        payload = {
            "question": question
        }
        res = _api(
            "POST",
            "/ask",
            json=payload
        )

        if res.status_code == 200:
            data = res.json()
            st.markdown("### Answer")
            st.write(
                data["answer"]
            )

def _tab_summary(filenames, page):
    if st.button("Summarize"):
        res = _api(
            "POST",
            "/summarize",
            json={}
        )

        if res.status_code == 200:
            data = res.json()
            st.write(
                data["summary"]
            )

def _tab_quiz(filenames, page):
    if st.button("Generate Quiz"):
        res = _api(
            "POST",
            "/quiz",
            json={}
        )

        if res.status_code == 200:
            data = res.json()
            st.json(data)

def _tab_flashcards(filenames,page):
    if st.button("Generative Flashcard"):
        res=_api("POST","/flashcards",json={})

        if res.status_code==200:
            data=res.json()
            st.json(data)

def run():
    st.set_page_config(
        page_title="RAG Learning System",
        layout="wide"
    )
    st.markdown(
        GLOBAL_CSS,
        unsafe_allow_html=True
    )
    filenames,page=_sidebar()

    tabs=st.tabs([
        "Hỏi đáp",
        "Tóm tắt",
        "Quiz",
        "Flashcards"
    ])
    handlers=[_tab_chat,_tab_summary,_tab_quiz,_tab_flashcards]

    for tab,fn in zip(tabs,handlers):
        with tab:
            fn(filenames,page)

if __name__ == "__main__":
    run()       