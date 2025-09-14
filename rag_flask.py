from flask import Flask, request, jsonify, render_template_string
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain.prompts import ChatPromptTemplate

# DB Path
DB_PATH = r"db\chroma_db"

# Flask App
app = Flask(__name__)

# Simple HTML Template (with a spinner for "progress")
HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>RAG Flask App</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        #answer { margin-top: 20px; font-size: 18px; }
        #loading { display: none; color: green; }
    </style>
</head>
<body>
    <h1>Ask a Question</h1>
    <form id="query-form">
        <input type="text" id="question" name="question" placeholder="Type your question..." size="60" required>
        <button type="submit">Ask</button>
    </form>
    <p id="loading">⏳ Processing... please wait</p>
    <div id="answer"></div>

    <script>
        const form = document.getElementById("query-form");
        const loading = document.getElementById("loading");
        const answerDiv = document.getElementById("answer");

        form.addEventListener("submit", async (e) => {
            e.preventDefault();
            answerDiv.innerHTML = "";
            loading.style.display = "block";

            const q = document.getElementById("question").value;
            const res = await fetch("/query", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ question: q })
            });

            const data = await res.json();
            loading.style.display = "none";
            answerDiv.innerHTML = "<b>Answer:</b> " + data.answer;
        });
    </script>
</body>
</html>
"""

@app.route("/")
def home():
    return render_template_string(HTML_PAGE)

@app.route("/query", methods=["POST"])
def query():
    data = request.json
    question = data.get("question")

    embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")
    vectordb = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)

    retriever = vectordb.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(question)
    context = "\n\n".join([doc.page_content for doc in docs])

    llm = OllamaLLM(model="mistral:latest")

    prompt = ChatPromptTemplate.from_template(
        "You are an assistant. Use the context below to answer the question.\n\n"
        "Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )

    final_prompt = prompt.format(context=context, question=question)
    response = llm.invoke(final_prompt)

    return jsonify({"answer": response})

if __name__ == "__main__":
    app.run(debug=True)
