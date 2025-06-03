
import os
import streamlit as st
from bs4 import BeautifulSoup
import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from dotenv import load_dotenv
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.prompts import PromptTemplate

load_dotenv()

def extract_text_from_url(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        return soup.get_text()
    except Exception as e:
        return f"Error fetching the URL: {e}"
    
def generate_answer(question, relevant_docs):
    context = "\n\n".join([doc.page_content for doc in relevant_docs])

    template = """
    You are a helpful assistant. Use the following retrieved documents to answer the user's question.

    Documents:
    {context}

    Question:
    {question}

    Answer:"""

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=template,
    )
    formatted_prompt = prompt.format(context=context, question=question)
    llm = ChatOpenAI(model='gpt-4',temperature=0.2)  # or use any other LLM

    response = llm.invoke(formatted_prompt)

    return response.content.strip()



def main():
    st.title("PDF QA System")
    st.write("Upload a PDF file to ask questions.")

    option = st.radio("Choose an input type:", ("PDF File"))

    if option == "PDF File":
        uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
        if uploaded_file is not None:
            # 1. Create 'uploaded_pdfs' folder if it doesn't exist
            upload_folder = "uploaded_pdfs"
            os.makedirs(upload_folder, exist_ok=True)

            # 2. Build the path where we'll save this PDF
            save_path = os.path.join(upload_folder, uploaded_file.name)

            # 3. Write the uploaded file to disk
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # 4. Now use PyPDFLoader on the saved file path
            loader = PyPDFLoader(save_path)
            docs = loader.load()

            # 5. Split the text into manageable chunks
            text_splitter = SemanticChunker(
                OpenAIEmbeddings(), breakpoint_threshold_type="standard_deviation",
                breakpoint_threshold_amount=3
            )
            # text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
            docs = text_splitter.split_documents(docs)
            
            # Store the list of Document objects in session state
            st.session_state["document_text"] = docs


            # create vector store from the documents
            vector_store = Chroma(
                embedding_function=OpenAIEmbeddings(),
                persist_directory='my_chroma_db',
                collection_name='sample'
            )
            
            # Save the vector store to disk
            ids = vector_store.add_documents(docs)

            

            print(f"Added {len(ids)} documents to the vector store.")

            st.success("PDF content loaded and processed successfully!")

    # elif option == "Website URL":
    #     url = st.text_input("Enter a website URL")
    #     if url:
            text = extract_text_from_url(url)
            st.session_state["document_text"] = text
            st.success("Website content loaded successfully!")

    # Show QA input once we have something in session_state
    if "document_text" in st.session_state and st.session_state["document_text"]:
        question = st.text_input("Ask a question about the content:")
        if question:
            # Create a retriever with compression
            embedding_model = OpenAIEmbeddings()
            base_retriever = vector_store.as_retriever(search_kwargs={"k": 3})
            llm = ChatOpenAI(model="gpt-3.5-turbo")
            compressor = LLMChainExtractor.from_llm(llm)
            compression_retriever = ContextualCompressionRetriever(
                base_retriever=base_retriever,
                base_compressor=compressor
            )
            # Use the retriever to get relevant documents
            relevant_docs = compression_retriever.invoke(question)

            # for doc in relevant_docs:
            #     print(doc.page_content)
            #     print()
            #     print("-------------------------------END OF PAGE-------------------------------")
            #     print()

            st.write("You asked:", question)
            
            generated_answer = generate_answer(question, relevant_docs)
                        
            st.write("Answer:", generated_answer)

if __name__ == "__main__":
    main()
