from flask import Flask, request, jsonify
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load and process PDF
loader = PyPDFLoader("document/Company_Policy.pdf")
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# Use a compatible embedding model
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Build vector store and retrieval QA
vectordb = Chroma.from_documents(chunks, embedding=embedding)
llm = Ollama(model="mistral:7b-instruct")
qa = RetrievalQA.from_chain_type(llm=llm, retriever=vectordb.as_retriever())

# Flask app
app = Flask(__name__)

@app.route("/ask", methods=["POST"])
def ask():
    query = request.json.get("query", "")
    if not query.strip():
        return jsonify({"error": "Query is empty"}), 400
    response = qa.run(query)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
