import os
import streamlit as st

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from langchain_groq import ChatGroq

load_dotenv()


## Uncomment the following files if you're not using pipenv as your virtual environment manager
#from dotenv import load_dotenv, find_dotenv
#load_dotenv(find_dotenv())


DB_FAISS_PATH="vectorstore/db_faiss"
@st.cache_resource
def get_vectorstore():
    embedding_model=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db=FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db


def set_custom_prompt(custom_prompt_template):
    prompt=PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt


def load_llm(huggingface_repo_id, HF_TOKEN):
    llm=HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.5,
        model_kwargs={"token":HF_TOKEN,
                      "max_length":"512"}
    )
    return llm


def main():
    st.title("Medical Chatbot!")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.markdown(message['content'])

    prompt=st.chat_input("Pass your prompt here")

    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role':'user', 'content': prompt})


        CUSTOM_PROMPT_TEMPLATE = """
        You are a medical information assistant.

        Answer the user's question using ONLY the information provided
        in the context.

        If the context contains relevant information, summarize it clearly
        and combine information from multiple context passages when appropriate.

        If the context does not contain enough information to answer the
        question, say that the provided document does not contain enough
        information.

        Do not invent medical facts or treatments that are not present
        in the context.

        Context:
        {context}

        Question:
        {question}

        Answer:
        """
                
        #HUGGINGFACE_REPO_ID="mistralai/Mistral-7B-Instruct-v0.3" # PAID
        #HF_TOKEN=os.environ.get("HF_TOKEN")  

        #TODO: Create a Groq API key and add it to .env file
        
        try: 
            vectorstore=get_vectorstore()
            if vectorstore is None:
                st.error("Failed to load the vector store")

            qa_chain = RetrievalQA.from_chain_type(
                llm=ChatGroq(
                    model_name="llama-3.3-70b-versatile",
                    temperature=0.0,
                    groq_api_key=st.secrets["GROQ_API_KEY"],
                ),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                return_source_documents=True,
                chain_type_kwargs={
                    'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)
                }
            )

            response = qa_chain.invoke({'query': prompt})

            result = response["result"]
            source_documents = response["source_documents"]

            # Show answer
            st.chat_message('assistant').markdown(result)

            # Show sources separately
            with st.expander("Sources"):
                for doc in source_documents:
                    st.write(
                        f"Page: {doc.metadata.get('page', 'Unknown')}"
                    )
                    st.write(doc.page_content)

            # Save only the answer in chat history
            st.session_state.messages.append({
                'role': 'assistant',
                'content': result
            })
            #response="Hi, I am MediBot!"
            # st.chat_message('assistant').markdown(result)
            # st.session_state.messages.append({'role':'assistant', 'content': result})

        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
