import Code from "../../components/Code/Code.jsx";
import img01 from "./img01.png"

const Content022 = () => {
  return (
    <div className="page container mt-5 mx-auto w-75 px-5 ">
      <h1 className="text-center">World Wars Q&A AI APP</h1>
      
      <p>
        In this project, we aim to build a sophisticated Retrieval-Augmented
        Generation (RAG) system using various tools and technologies that can
        answer questions related to world wars. The project involves several key
        steps, from setting up the language model to deploying the application.
        The entire system is integrated with Streamlit, leveraging its Session
        State feature to maintain and share variables between reruns, ensuring a
        seamless user experience. Finally, the Streamlit application is deployed
        to the cloud.
      </p>
      <a                
                    href={"https://warsqna-oagzovdnn3stumkdepqhsd.streamlit.app/"} target="_blank"
                    className="project-link m-0 texr-decoration-underline"
                  >Streamlit Cloud link to the App</a>
      <h5>RAG Framework</h5>
      <Code
        code={`
          #Step1: LLM model
          from langchain_groq import ChatGroq
          load_dotenv()
          os.environ['HF_TOKEN']=st.secrets['HF_TOKEN']
          groq_api_key = os.getenv('GROQ_API_KEY')
          llm_model = ChatGroq(model='Llama3-8b-8192',groq_api_key=groq_api_key)

          #Step2: RAG Implementation
          #Data injection
          from langchain_community.document_loaders import PyPDFDirectoryLoader
          loader=PyPDFDirectoryLoader("wars_pdf")
          docs=loader.load()
          #Data chunking
          from langchain.text_splitter import RecursiveCharacterTextSplitter
          text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=100)
          final_documents=text_splitter.split_documents(docs)
          #Documnet Embedding
          from langchain_huggingface import HuggingFaceEmbeddings
          embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
          #FAISS Vector store
          from langchain_community.vectorstores import FAISS
          vector_store=FAISS.from_documents(final_documents,embeddings)
          
          `}
      />
      <p>
        we import the ChatGroq class from the langchain_groq module. This class
        is likely used to interact with a language model provided by Groq. The
        load_dotenv() function is called to load environment variables from a
        .env file, which is a common practice to manage sensitive information
        like API keys and tokens. The os.environ['HF_TOKEN'] line sets the
        Hugging Face token from the streamlit secrets, ensuring secure access to
        the Hugging Face API.
        <br />
        Next, the code retrieves the Groq API key from the environment variables
        using os.getenv('GROQ_API_KEY'). This key is essential for
        authenticating requests to the Groq API. Finally, the llm_model variable
        is initialized with an instance of ChatGroq, specifying the model name
        Llama3-8b-8192 and the Groq API key. This setup prepares the language
        model for further use in the application.
        <br />
        <p>
          we import the PyPDFDirectoryLoader class from the
          langchain_community.document_loaders module. This class is used to
          load PDF documents from a specified directory, in this case,
          “wars_pdf”. The loaded documents are stored in the docs variable.
          Next, we use the RecursiveCharacterTextSplitter from the
          langchain.text_splitter module to split the loaded documents into
          smaller chunks. This is done to manage large documents more
          effectively. The chunk_size is set to 1000 characters with an overlap
          of 100 characters, ensuring that the chunks are manageable and that
          important information isn’t lost between chunks. The resulting chunks
          are stored in the final_documents variable.
          <br />
          We then imports the HuggingFaceEmbeddings class from the
          langchain_huggingface module to create embeddings for the document
          chunks. The model used for this purpose is “all-MiniLM-L6-v2”, which
          is known for its efficiency in generating embeddings. Finally, we
          import the FAISS class from the langchain_community.vectorstores
          module to create a vector store from the document embeddings. This
          vector store is used to efficiently search and retrieve relevant
          document chunks based on their embeddings.
          <br />
        </p>
      </p>
      <h5>Prompts|chains|Retrevials</h5>
      <Code
        code={`
          #step:Prompts|Chains|retreival
          from langchain_core.prompts import ChatPromptTemplate
          prompt=ChatPromptTemplate.from_template(
              """
              Answer the questions based on the provided context only.
              Please provide the most accurate respone based on the question
              <context>
              {context}
              <context>
              Question:{input}

              """

          )
          from langchain.chains.combine_documents import create_stuff_documents_chain
          from langchain.chains import create_retrieval_chain
          document_chain=create_stuff_documents_chain(llm_model,prompt)
          retriever=vector_store.as_retriever()
          retrieval_chain=create_retrieval_chain(retriever,document_chain)
          response=retrieval_chain.invoke({'input':"How many countries involved in world war 1 and their names"})
          response['answer']
          
          `}
      />
      <p>
        We import the ChatPromptTemplate class from the langchain_core.prompts
        module. This class is used to create a prompt template for the language
        model. The prompt variable is initialized with a template that instructs
        the model to answer questions based on the provided context only,
        ensuring accurate responses. Next, the code imports functions to create
        document and retrieval chains. The create_stuff_documents_chain function
        is used to create a chain that combines documents using the language
        model and the prompt template. This chain is stored in the
        document_chain variable. The vector_store is converted into a retriever
        using the as_retriever method. This retriever is then used to create a
        retrieval chain with the create_retrieval_chain function, which combines
        the retriever and the document chain. The resulting retrieval chain is
        stored in the retrieval_chain variable.
        <br />
      </p>
      <h4>Streamlit APP</h4>
      <Code
        code={`
          #Step4: Streamlit interface complete app

          import os
          from dotenv import load_dotenv
          from langchain_groq import ChatGroq
          from langchain_core.messages import HumanMessage
          from langchain_community.document_loaders import PyPDFDirectoryLoader
          from langchain.text_splitter import RecursiveCharacterTextSplitter
          from langchain_huggingface import HuggingFaceEmbeddings
          from langchain_community.vectorstores import FAISS
          from langchain_core.prompts import ChatPromptTemplate
          from langchain.chains.combine_documents import create_stuff_documents_chain
          from langchain.chains import create_retrieval_chain
          import streamlit as st


          load_dotenv()
          os.environ['HF_TOKEN']=st.secrets['HF_TOKEN']
          groq_api_key = st.secrets['GROQ_API_KEY']

          #Step1: LLM model

          llm_model = ChatGroq(model='Llama3-8b-8192',groq_api_key=groq_api_key)
          #llm_model.invoke([HumanMessage(content="tell me joke")])

          #Step2: RAG implementation
          # Data injection and chunking
          #Document Embedding and Vectore stores
          def create_vectore_store():
              if "vectore_store" not in st.session_state:
                  st.session_state.embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                  st.session_state.loader = PyPDFDirectoryLoader("wars_pdf")
                  st.session_state.docs=st.session_state.loader.load()
                  st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=100)
                  st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs)
                  st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings)

          st.title("World wars Q&A with Llama3 based AI")
          st.write("First Press start and wait for AI to be trainied")
          st.write("Trainied on only small data of world wars.")
          user_prompt = st.text_input("Enter question regarding world wars")

          if st.button("Start"):
              create_vectore_store()
              st.write("Trainied and ready, please enter questions!")

          #Step3: Chains and Retrevials
          prompt=ChatPromptTemplate.from_template(
              """
              Answer the questions based on the provided context only.
              Please provide the most accurate respone based on the question
              <context>
              {context}
              <context>
              Question:{input}

              """

          )

          if user_prompt:
              document_chain=create_stuff_documents_chain(llm_model,prompt)
              retriever=st.session_state.vectors.as_retriever()
              retrieval_chain=create_retrieval_chain(retriever,document_chain)

              response = retrieval_chain.invoke({'input': user_prompt})

              st.write(response['answer'])

              with st.expander("Document similarity Search"):
                  for i,doc in enumerate(response['context']):
                      st.write(doc.page_content)
                      st.write('------------------------')          
          `}
      />
      <p>
        1. st.session_state: Streamlit’s Session State is a feature that allows
        you to maintain and share variables between reruns of your Streamlit
        app, ensuring a more interactive and dynamic user experience. Each time
        a user interacts with your app, Streamlit reruns the script from top to
        bottom. Without Session State, variables would reset with each rerun,
        losing any user-specific data.Session State enables you to store and
        persist data across these reruns for each user session. In our case we
        implement al the RAG frame work only once at the start and we use same
        vector store for all refreshs of page, which in turn saves time.
        2.user_promt: We get question from user as user_promt variable and
        create chains prompts and retreivers accordingly and output the
        appropriate result.
        <br />
      </p>
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

export default Content022;
