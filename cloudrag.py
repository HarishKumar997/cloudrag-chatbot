from flask import Flask, request, jsonify
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

# === Load and Split the PDF ===
loader = PyPDFLoader("document/Company_Policy.pdf")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# === HuggingFace Embeddings with trust_remote_code ===
embedding = HuggingFaceEmbeddings(
    model_name="nomic-ai/nomic-embed-text-v1",
    model_kwargs={"trust_remote_code": True}
)

# === Create Vector Store ===
vectordb = Chroma.from_documents(chunks, embedding=embedding)

# === Load Local Ollama LLM ===
llm = Ollama(model="mistral:7b-instruct")

# === Setup RetrievalQA Chain ===
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectordb.as_retriever()
)

# === Flask App Setup ===
app = Flask(__name__)

@app.route("/ask", methods=["POST"])
def ask():
    query = request.json.get("query")

    if not query or not isinstance(query, str) or not query.strip():
        return jsonify({"error": "Invalid query"}), 400

    response = qa.run(query)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
