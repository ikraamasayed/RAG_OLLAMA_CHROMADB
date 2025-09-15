from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain.prompts import ChatPromptTemplate
from config import DB_PATH

def get_answer(question: str) -> str:
    try:
        embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")
        vectordb = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)

        retriever = vectordb.as_retriever(search_kwargs={"k": 3})
        docs = retriever.invoke(question)
        context = "\n\n".join([doc.page_content for doc in docs]) if docs else "No relevant documents found."

        llm = OllamaLLM(model="mistral:latest")

        prompt = ChatPromptTemplate.from_template(
            "You are an assistant. Use the context below to answer the question.\n\n"
            "Context:\n{context}\n\nQuestion: {question}\nAnswer:"
        )

        final_prompt = prompt.format(context=context, question=question)
        response = llm.invoke(final_prompt)
        return response
    except Exception as e:
        return f"⚠️ Error: {str(e)}"
