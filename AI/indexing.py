from dotenv import load_dotenv
from langchain_community.document_loaders import RecursiveUrlLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

loader = RecursiveUrlLoader("https://nextjs.org/docs")
print("loading...")
docs = loader.load()

# print(docs[0].metadata.get("source"))

allLoader = WebBaseLoader(web_paths = [docs[i].metadata.get("source") for i in range(len(docs))])
print("loading all..")
allDocs = allLoader.load()

split_text = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=300
)
print("spliting...")
split = split_text.split_documents(documents=allDocs)

embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
print("creating vector...")
vector = QdrantVectorStore.from_documents(
    documents=split,
    embedding=embedding_model,
    url="http://localhost:6333",
    collection_name="nextjsVector"
)
print("vector created")

