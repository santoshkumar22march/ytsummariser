import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up Groq API
my_secret = os.environ['GROQ_API_KEY']
llm = ChatGroq(api_key=my_secret, model_name="mixtral-8x7b-32768")

def get_video_transcript(video_id):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return " ".join([entry['text'] for entry in transcript])
    except Exception as e:
        st.error(f"Error fetching transcript: {str(e)}")
        return None

def summarize_text(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)

    summary_template = """
    Summarize the following text in a concise and informative manner:

    {text}

    Summary:
    """

    prompt = PromptTemplate(template=summary_template, input_variables=["text"])

    summarization_chain = LLMChain(llm=llm, prompt=prompt)

    summaries = []
    for chunk in chunks:
        summary = summarization_chain.run(chunk)
        summaries.append(summary)

    return " ".join(summaries)

def main():
    st.title("YouTube Video Summarizer")

    video_url = st.text_input("Enter YouTube Video URL:")

    if video_url:
        video_id = video_url.split("v=")[1] if "v=" in video_url else video_url.split("/")[-1]

        if st.button("Summarize"):
            with st.spinner("Fetching transcript..."):
                transcript = get_video_transcript(video_id)

            if transcript:
                with st.spinner("Generating summary..."):
                    summary = summarize_text(transcript)
                    st.subheader("Video Summary")
                    st.write(summary)
            else:
                st.error("Failed to fetch the transcript. Please check the video URL and try again.")

if __name__ == "__main__":
    main()
