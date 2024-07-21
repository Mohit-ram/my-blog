import Code from "../../components/Code/Code.jsx";
import img01 from "./img01.png";

const Content023 = () => {
  return (
    <div className="page container mt-5 mx-auto w-75 px-5 ">
      <h1 className="text-center">Contextual Q&A AI APP</h1>
      <p>
        In the rapidly evolving field of artificial intelligence, the ability to
        effectively retrieve and generate relevant information is crucial. This
        project focuses on creating a Conversational Retrieval-Augmented
        Generation (RAG) Chain to enhance user-AI interactions. By integrating
        advanced retrieval mechanisms with generative models, the system aims to
        provide accurate and contextually relevant responses. The key components
        include a Question-Answer Chain leveraging a language model (LLM), a
        Retrieval-Augmented Generation (RAG) Chain that combines retrieval
        systems with generative models, and robust session management to
        maintain conversation continuity. And also using streamlit interface the AI is implemented and deployed in cloud.
      </p>
      <a
        href={"https://contextualqnaai-k4kibvvy8vspmxwbsx9vfv.streamlit.app/"}
        target="_blank"
        className="project-link m-0 texr-decoration-underline"
      >
        Streamlit Cloud link to the App
      </a>
      <h5>RAG Framework</h5>
      <Code
        code={`
          #Step1: AI Model
          from dotenv import load_dotenv
          load_dotenv()
          os.environ['HF_TOKEN']=os.getenv("HF_TOKEN")
          groq_api_key = os.getenv('GROQ_API_KEY')
          embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
          llm=ChatGroq(groq_api_key=groq_api_key,model_name="Llama3-8b-8192")

          #Step2: RAG Chain
          #Data Loadin, Data chunking, Documnet Embedding, VectorStores
          loader=PyPDFLoader('test.pdf')
          documents=loader.load()
          text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=200)
          splits = text_splitter.split_documents(documents)
          vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
          retriever = vectorstore.as_retriever() 
          `}
      />
      <p>
        In the first step, we use the dotenv library to load environment
        variables from a .env file. This is a secure way to manage sensitive
        information like API keys. WE then set these environment variables for
        HuggingFace and Groq API keys. The HuggingFace embeddings model
        (all-MiniLM-L6-v2) is initialized to convert text into numerical
        representations (embeddings). The ChatGroq language model
        (Llama3-8b-8192) is also initialized using the Groq API key.
        <br />
        In the second step, We load a PDF document using the PyPDFLoader class.
        This document is then split into smaller chunks using the
        RecursiveCharacterTextSplitter, which ensures that each chunk is of a
        manageable size (5000 characters) with a slight overlap (200 characters)
        to maintain context. These chunks are then embedded using the
        HuggingFace embeddings model and stored in a FAISS (Facebook AI
        Similarity Search) vector store. This vector store is configured to act
        as a retriever, allowing efficient retrieval of relevant document chunks
        based on a query.
        <br />
      </p>
      <h5>Prompts|chains|Retrevials</h5>
      <Code
        code={`
          # Step 3: Prompts and history aware
          # Chat history aware prompt

          # System message for reformulating questions based on chat history
          system_with_history_message = (
              "Given a chat history and the latest user question"
              " which might reference context in the chat history, "
              "formulate a standalone question which can be understood "
              "without the chat history. Do NOT answer the question, "
              "just reformulate it if needed and otherwise return it as is."
          )

          # Create a chat prompt template with the system message and placeholders
          system_with_history_prompt = ChatPromptTemplate.from_messages(
              [
                  ("system", system_with_history_message),
                  MessagesPlaceholder("chat_history"),
                  ("human", "{input}"),
              ]
          )

          # Create a history-aware retriever by combining LLM, retriever, and the history-aware prompt
          history_aware_retriever = create_history_aware_retriever(llm, retriever, system_with_history_prompt)

          # Actual system prompt for question-answering tasks
          system_actual_prompt = (
              "You are an assistant for question-answering tasks. "
              "Use the following pieces of retrieved context to answer "
              "the question. If you don't know the answer, say that you "
              "don't know. Use three sentences maximum and keep the "
              "answer concise."
              "\n\n"
              "{context}"
          )

          # Create a chat prompt template for the actual system prompt
          qa_prompt = ChatPromptTemplate.from_messages(
              [
                  ("system", system_actual_prompt),
                  MessagesPlaceholder("chat_history"),
                  ("human", "{input}"),
              ]
          )

          `}
      />
      <p>
        In this step, we create prompts that are aware of the chat history to
        ensure that questions are contextually accurate and relevant. First, we
        set up a system message that instructs the model to reformulate a user’s
        question into a standalone question that can be understood without the
        chat history. This is achieved using the ChatPromptTemplate with a
        system message and placeholders for chat history and user input.
        <br />
        Next, we create a history-aware retriever by combining the language
        model (LLM) and the retriever with the history-aware prompt. This
        ensures that the retriever can handle context from the chat history
        effectively. Finally, we set up the actual system prompt for
        question-answering tasks. This prompt instructs the assistant to use
        retrieved context to answer questions concisely. The ChatPromptTemplate
        is used again to structure this prompt with placeholders for chat
        history and user input.
        <br />
        <h4>Runnable chains</h4>
        <Code
          code={`
                    # Step 4: Runnable chains
          # Create the question-answer chain
          question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

          # Create the Retrieval-Augmented Generation (RAG) chain
          rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
          # Session management
          store = {}
          # Function to get session history
          def get_session_history(session_id: str) -> BaseChatMessageHistory:
              if session_id not in store:
                  store[session_id] = ChatMessageHistory()
              return store[session_id]

          # Create the conversational RAG chain
          conversational_rag_chain = RunnableWithMessageHistory(
              rag_chain,
              get_session_history,
              input_messages_key="input",
              history_messages_key="chat_history",
              output_messages_key="answer"
          )
          # Invoke the chain with a sample input and configuration
          response = conversational_rag_chain.invoke(
              {"input": "what is the topic about?"},
              config={
                  "configurable": {"session_id": "chat1"}
              }  
          )
          # Retrieve the answer from the response
          response['answer']

`}
        />
        <p>
          In this step, we focus on creating runnable chains to handle the
          question-answering process effectively. We begin by setting up a
          question_answer_chain using the create_stuff_documents_chain function.
          This chain leverages the language model (LLM) and the
          question-answering prompt (qa_prompt) to generate answers based on the
          provided context. This ensures that the assistant can produce accurate
          and relevant responses. Next, we create a rag_chain using the
          create_retrieval_chain function. This chain integrates the
          history-aware retriever with the question-answer chain, allowing for
          contextually relevant answers by considering the entire conversation
          history. This step is crucial for maintaining the flow and coherence
          of the conversation, ensuring that the assistant can retrieve and
          utilize relevant information effectively.
          <br />
          We then manage session histories using a store dictionary. The
          get_session_history function retrieves the chat history for a given
          session ID, initializing a new ChatMessageHistory object if the
          session ID does not exist in the store. This setup allows us to
          maintain context across different sessions, enhancing the assistant’s
          ability to provide consistent and context-aware responses.Finally, we
          create a conversational_rag_chain using the RunnableWithMessageHistory
          class. This chain combines the RAG chain with the session history
          retrieval function, specifying keys for input messages, chat history,
          and output messages. By invoking this chain with a sample input
          question and a configuration that includes a session ID, we ensure
          that the assistant can handle conversations with context awareness.
          This setup allows the assistant to provide accurate and concise
          answers based on the chat history and retrieved context, enhancing the
          overall user experience.
          <br />
        </p>
      </p>
      <h4>Streamlit APP</h4>
      <Code
        code={`
        
              import streamlit as st
              from langchain.chains import create_history_aware_retriever, create_retrieval_chain
              from langchain.chains.combine_documents import create_stuff_documents_chain
              from langchain_community.vectorstores import FAISS
              from langchain_community.chat_message_histories import ChatMessageHistory
              from langchain_core.chat_history import BaseChatMessageHistory
              from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
              from langchain_groq import ChatGroq
              from langchain_core.runnables.history import RunnableWithMessageHistory
              from langchain_huggingface import HuggingFaceEmbeddings
              from langchain_text_splitters import RecursiveCharacterTextSplitter
              from langchain_community.document_loaders import PyPDFLoader
              import os

              #Step1: AI Model
              from dotenv import load_dotenv
              load_dotenv()
              os.environ['HF_TOKEN']=os.getenv("HF_TOKEN")
              groq_api_key = os.getenv('GROQ_API_KEY')

              embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
              llm=ChatGroq(groq_api_key=groq_api_key,model_name="Llama3-8b-8192")

              #session initalisation
              st.title("Contextual Q&A with Chat History ")
              st.write("Upload Pdf's and ask about them")
              st.write("AI can only respond regarding the context and amount of context provide.")
              session_id=st.text_input("Session ID",value="default_session")
              if 'session_store' not in st.session_state:
                      st.session_state.session_store={}

              #Step2: RAG Impementtaion
                  #data loadig
              uploaded_files=st.file_uploader("Choose A PDf file",type="pdf",accept_multiple_files=True)
              if uploaded_files:
                  documents =[]
                  for uploaded_file in uploaded_files:
                      temppdf = f"./temp.pdf"
                      with open(temppdf, 'wb') as file:
                              file.write(uploaded_file.getvalue())
                              file_name = uploaded_file.name

                      pdf_read=PyPDFLoader(temppdf).load()
                      documents.extend(pdf_read)
                  #data Chunking and vector stores
                  text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=200)
                  splits = text_splitter.split_documents(documents)
                  vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
                  retriever = vectorstore.as_retriever() 

              #Step3:Prompts 
                  # History aware prompt
                  system_with_history_message=(
                              "Given a chat history and the latest user question"
                              "which might reference context in the chat history, "
                              "formulate a standalone question which can be understood "
                              "without the chat history. Do NOT answer the question, "
                              "just reformulate it if needed and otherwise return it as is."
                          )
                  system_with_history_prompt = ChatPromptTemplate.from_messages(
                                  [
                                      ("system", system_with_history_message),
                                      MessagesPlaceholder("chat_history"),
                                      ("human", "{input}"),
                                  ]
                              )
                  history_aware_retriever=create_history_aware_retriever(llm,retriever,system_with_history_prompt)
                  # actual QNA prompt
                  system_actual_prompt = (
                                  "You are an assistant for question-answering tasks. "
                                  "Use the following pieces of retrieved context to answer "
                                  "the question. If you don't know the answer, say that you "
                                  "don't know. Use three sentences maximum and keep the "
                                  "answer concise."
                                  "\n\n"
                                  "{context}"
                              )
                  qa_prompt = ChatPromptTemplate.from_messages(
                                  [
                                      ("system", system_actual_prompt),
                                      MessagesPlaceholder("chat_history"),
                                      ("human", "{input}"),
                                  ]
                              )
                  
                  #Step4: Runnable chains
                  question_answer_chain=create_stuff_documents_chain(llm,qa_prompt)
                  rag_chain=create_retrieval_chain(history_aware_retriever,question_answer_chain)
                  
                  def get_session_history(session_id:str)->BaseChatMessageHistory:
                          if session_id not in st.session_state.session_store:
                              st.session_state.session_store[session_id]=ChatMessageHistory()
                          return st.session_state.session_store[session_id]
                  
                  conversational_rag_chain=RunnableWithMessageHistory(
                          rag_chain,get_session_history,
                          input_messages_key="input",
                          history_messages_key="chat_history",
                          output_messages_key="answer"
                      )
                  user_input = st.text_input("Your question:")
                  if user_input:
                      session_history=get_session_history(session_id)
                      response = conversational_rag_chain.invoke(
                          {"input": user_input},
                          config={
                              "configurable": {"session_id":session_id}
                          },
                      )
                      st.write(st.session_state.session_store)            
                      st.write("Assistant:", response['answer'])
                      st.write("Chat History:", session_history.messages)
`}
      />
      <h4>Deployment in Sreamlit cloud</h4>
      <br />
      1. Push current directory files (app.py, requirements, pdf_folder) to a
      new git repository
      <br />
      2. Commit changes and loign streamlit cloud and connect with github.
      <br />
      2. Go to Streamlit cloud then "create app" select git repo, under advanced
      setting paste GROQ api key as secret key (groq_api_key="gq-ddsa...")
      <br />
      3. save and press deploy. and wait until app is build.
      <div className="text-center">
        <p>Deployed App</p>
        <img
          src={img01}
          alt="result1"
          style={{ height: "400px", width: "600px" }}
        />
      </div>
    </div>
  );
};

export default Content023;
