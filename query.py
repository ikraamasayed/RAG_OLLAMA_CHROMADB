import os
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain.prompts import ChatPromptTemplate


DB_PATH = r"db\chroma_db" 

def query_rag(question):
    
    print('... Querying RAG system ...\n')

    # Load DB
    print('... loading embeddings ...')
    embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")
    print('... Loading vector database ...')
    vectordb = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)

    print('... Retrieving revelent documnets ...')
    # Retrieve relevant docs
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})
    # docs = retriever.get_relevant_documents(question) # old method
    print('... retrierver.invoke() ...')
    docs = retriever.invoke(question)

    print(f"Retrieved {len(docs)} documents context.\n")
    context = "\n\n".join([doc.page_content for doc in docs])

    # LLM
    print('... Selection of LLM model ...')
    llm = OllamaLLM(model="mistral:latest")  # you can try gemma3/qwen3 as well

    # Prompt
    print('... Preparing Prompt ...')
    prompt = ChatPromptTemplate.from_template(
        "You are an assistant. Use the context below to answer the question.\n\n"
        "Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )

    print('... prompt fromatting (context + question) ...')
    final_prompt = prompt.format(context=context, question=question)
    print('... llm.invoke() ...')
    response = llm.invoke(final_prompt)

    print(f'You asked: {question}\n')
    print("---- Answer ----")
    print(response)

if __name__ == "__main__":
    while True:
        q = input("Ask: ")
        if q.lower() in ["exit", "quit"]:
            break
        query_rag(q)
