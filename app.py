import re
import validators
import streamlit as st

from youtube_transcript_api import YouTubeTranscriptApi
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_groq import ChatGroq

#-------------Page Design----------------
st.markdown("""
<style>

/* --------- Main Background --------- */
.stApp {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: white;
}

/* --------- Glass Container --------- */
.main > div {
    background-color: rgba(255, 255, 255, 0.05);
    padding: 2rem;
    border-radius: 15px;
    backdrop-filter: blur(10px);
}

/* --------- Title --------- */
h1, h2, h3 {
    color: #ffffff;
    text-align: center;
}

/* --------- Text Inputs --------- */
.stTextInput > div > div > input {
    background-color: #1e1e1e;
    color: white;
    border-radius: 8px;
    border: 1px solid #555;
    padding: 10px;
}

/* --------- Button --------- */
.stButton button {
    background: linear-gradient(90deg, #ff512f, #dd2476);
    color: white;
    border: none;
    border-radius: 8px;
    height: 3em;
    width: 100%;
    font-size: 16px;
    font-weight: bold;
    transition: 0.3s;
}

.stButton button:hover {
    transform: scale(1.03);
    box-shadow: 0px 0px 15px rgba(255, 81, 47, 0.6);
}

/* --------- Success Box --------- */
.stAlert {
    border-radius: 10px;
}

/* --------- Hide Streamlit Footer --------- */
footer {visibility: hidden;}

</style>
""", unsafe_allow_html=True)

# -------- YouTube Transcript Loader (fixed) --------
def get_youtube_docs(url: str):
    video_id = re.search(r"(?:v=|youtu\.be/)([^&]+)", url).group(1)
    api = YouTubeTranscriptApi()
    transcript = api.fetch(video_id)
    text = " ".join([item.text for item in transcript])
    return [Document(page_content=text)]


# -------- UI --------
st.set_page_config(page_title="YT / Website Summarizer", page_icon="🎬")
st.title("🎬 YouTube / Website Summarizer")

with st.sidebar:
    groq_api_key = st.text_input("Groq API Key", type="password")
url = st.text_input("Enter YouTube or Website URL")

if st.button("Summarize"):

    if not groq_api_key or not validators.url(url):
        st.error("Provide valid Groq API key and URL")
        st.stop()

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        groq_api_key=groq_api_key,
        temperature=0
    )

    prompt = PromptTemplate.from_template("""
    Provide a clear 300-word summary of the following content.

    Content:
    {context}
    """)

    chain = prompt | llm | StrOutputParser()

    try:
        with st.spinner("Loading content..."):

            if "youtube.com" in url or "youtu.be" in url:
                docs = get_youtube_docs(url)
            else:
                loader = UnstructuredURLLoader(urls=[url])
                docs = loader.load()

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=2000,
                chunk_overlap=200
            )
            split_docs = splitter.split_documents(docs)

            full_text = "\n\n".join([doc.page_content for doc in split_docs])

            summary = chain.invoke({"context": full_text})

            st.success("Summary")
            st.success(summary)

    except Exception as e:
        st.exception(e)