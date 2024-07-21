import Code from "../../components/Code/Code.jsx";
import img01 from "./img01.png";

const Content024 = () => {
  return (
    <div className="page container mt-5 mx-auto w-75 px-5 ">
      <h1 className="text-center">WebPage summarization</h1>      
      <p>
        In today’s digital age, the sheer volume of information available online
        can be overwhelming. With countless articles, research papers, and web
        pages, it becomes increasingly challenging to extract key insights
        efficiently. This project aims to address this issue by developing a web
        application that leverages advanced natural language processing (NLP)
        techniques to summarize content from any given URL. Utilizing the
        capabilities of the Groq API and Streamlit, this application provides
        users with concise summaries of web content, enabling them to grasp
        essential information quickly. The project integrates various
        components, including URL validation, content extraction, and
        summarization chains, to deliver accurate and coherent summaries. By
        automating the summarization process, this tool not only saves time but
        also enhances productivity, making it an invaluable resource for
        researchers, students, and professionals alike.
      </p>
      <a
        href={"https://texsummarylc-zqmbbyxqndscfy3xr7nzyp.streamlit.app/"}
        target="_blank"
        className="project-link m-0 texr-decoration-underline"
      >
        Streamlit Cloud link to the App
      </a>
      <h4>Streamlit app</h4>
      <Code
        code={`
          import validators, streamlit as st
          from langchain.prompts import PromptTemplate
          from langchain_groq import ChatGroq
          from langchain.chains.summarize import load_summarize_chain
          from langchain_community.document_loaders import UnstructuredURLLoader
          import os

          # Streamlit app configuration
          st.set_page_config(page_title="Text Summarization")
          st.title("Text Summarization of a webpage")
          st.subheader('Summarize URL')

          # Get the Groq API Key and URL to be summarized
          groq_api_key = st.secrets["GROQ_API_KEY"]
          llm = ChatGroq(groq_api_key=groq_api_key, model="Gemma-7b-It")
          generic_url = st.text_input("URL", label_visibility="collapsed")

          # Define the prompt template for summarization
          prompt_template = """
          Provide a summary of the following content in 300 words:
          Content:{text}
          """
          prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

          # Button to trigger summarization
          if st.button("Summarize the Content from a Website"):
              # Validate inputs
              if not groq_api_key.strip() or not generic_url.strip():
                  st.error("Please provide the information to get started")
              elif not validators.url(generic_url):
                  st.error("Please enter a valid URL.")
              else:
                  try:
                      with st.spinner("Waiting..."):
                          # Load content from the URL
                          loader = UnstructuredURLLoader(
                              urls=[generic_url], 
                              ssl_verify=False,
                              headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"}
                          )
                          docs = loader.load()

                          # Chain for summarization
                          chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                          output_summary = chain.run(docs)

                          # Display the summary
                          st.success(output_summary)
                  except Exception as e:
                      st.exception(f"Exception: {e}")
       
          `}
      />
      <p>
        We begin by importing necessary libraries: validators for URL
        validation, streamlit for creating the web application, PromptTemplate
        from langchain.prompts for defining the prompt template, ChatGroq from
        langchain_groq for interacting with the Groq API, load_summarize_chain
        from langchain.chains.summarize for loading the summarization chain, and
        UnstructuredURLLoader from langchain_community.document_loaders for
        loading content from a URL. Additionally, the os module is imported for
        handling environment variables. Next, the Streamlit app is configured
        with a title and a subheader. The st.set_page_config function sets the
        page title to “Text Summarization,” and the st.title and st.subheader
        functions display the main title and subheader on the app’s interface.
        <br />
        The app then retrieves the Groq API key from Streamlit’s secrets
        management and initializes the ChatGroq object with the API key and
        model name. It also creates a text input field for the user to enter the
        URL of the content to be summarized. The PromptTemplate is defined with
        a template that instructs the model to provide a summary of the given
        content in 300 words. When the “Summarize the Content from a Website”
        button is clicked, the app validates the inputs. If the API key or URL
        is missing, or if the URL is invalid, appropriate error messages are
        displayed. If the inputs are valid, the app attempts to load the content
        from the URL using UnstructuredURLLoader. The loader is configured to
        bypass SSL verification and use a specific user-agent header.
        <br />
        The loaded content is then passed to the summarization chain, which is
        created using the load_summarize_chain function. The chain processes the
        content and generates a summary based on the defined prompt. The summary
        is displayed on the app’s interface using st.success. If any exceptions
        occur during this process, they are caught and displayed using
        st.exception.
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

export default Content024;
