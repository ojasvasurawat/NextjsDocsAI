from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from google import genai
import streamlit as st 

load_dotenv()

st.title("Next.js Docs AI")
st.write("We are here to help you with next.js documentations")

embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

query_vector = QdrantVectorStore.from_existing_collection(
    url="http://localhost:6333",
    collection_name="nextjsVector",
    embedding=embedding_model
)

SYSTEM_PROMPT = f"""
    You are a helpful AI assistant who answers user query based on the available context retrieved from a Website.
    along with page_contents and the source links and title and description from where you get the information.

    You should only answer the user based on the context given in the user query.
    and navigate the user to open the right link to know more.
    Answer the question step-by-step. 
    Answer should be detailed enabling user to understand the content better.
    Give long answers.
    If a user asks anything outside of the context, refrain from answering the question.

"""

client = genai.Client()

messages = []
messages.append({"role": "user", "parts": [{"text": SYSTEM_PROMPT}]})

# while True:
# query = input("user > ")
query = st.text_input(label="",label_visibility="collapsed", placeholder="Ask me a question about Next.js...")
search_result = query_vector.similarity_search(query=query)

context = "\n\n\n\n".join([
    f"Page Content: {result.page_content}\n Source: {result.metadata.get("source")}\n Title: {result.metadata.get("title")}\n Description: {result.metadata.get("description")}"
    for result in search_result])
messages.append(query + f"Context : {context}")
if query:
    # while True:
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=messages
    )
    messages.append(response.text)
    # print("🤖: ", response.text)
    st.write(response.text)
    # st.write(query)
    # break
