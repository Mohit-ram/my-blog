import Code from "../../components/Code/Code.jsx";


const Content025 = () => {
  return (
    <div className="page container mt-5 mx-auto w-75 px-5 ">
      <h1 className="text-center">YT Video summarization</h1>
      <p>
        In today’s digital age, video content is abundant, and extracting
        meaningful information from lengthy videos can be time-consuming. This
        project aims to address this challenge by leveraging artificial
        intelligence to summarize YouTube videos efficiently. By utilizing the
        LangChain library and the Groq API, we have developed a Streamlit web
        application that generates concise summaries of video content based on
        transcripts obtained from YouTube.
        <br />
        The application is designed to be user-friendly, allowing users to input
        a YouTube URL and receive a summary of the video’s content. This not
        only saves time but also provides a quick overview of the video’s key
        points, making it easier for users to decide whether to watch the full
        video. The integration of advanced AI models ensures that the summaries
        are accurate and informative, providing a valuable tool for content
        consumption in the digital era.
      </p>
      {/* <a
        href={"https://texsummarylc-zqmbbyxqndscfy3xr7nzyp.streamlit.app/"}
        target="_blank"
        className="project-link m-0 texr-decoration-underline"
      >
        Streamlit Cloud link to the App
      </a> */}
      <h4>Streamlit app</h4>
      <Code
        code={`
          import validators,streamlit as st
          from langchain.prompts import PromptTemplate
          from langchain_groq import ChatGroq
          from langchain.chains.summarize import load_summarize_chain
          from langchain_community.document_loaders import YoutubeLoader,UnstructuredURLLoader
          import os

          ## sstreamlit APP
          st.set_page_config(page_title="Youtube Video summary")
          st.title("Youtube Video summary")
          st.write("AI tries to summarise video based on transcripts obtainedf rom youtube.")
          st.write("More context more training time and more reasonable summary")
          st.subheader('Youtube URL')



          ## Get the Groq API Key and url(YT or website)to be summarized
          groq_api_key=st.secrets["GROQ_API_KEY"]
          llm=ChatGroq(groq_api_key=groq_api_key,model="Gemma-7b-It")
          generic_url=st.text_input("URL",label_visibility="collapsed")

          ## Gemma Model USsing Groq API
          llm =ChatGroq(model="Gemma-7b-It", groq_api_key=groq_api_key)

          prompt_template="""
          Provide a summary of the following content in 300 words:
          Content:{text}

          """
          prompt=PromptTemplate(template=prompt_template,input_variables=["text"])

          if st.button("Summarize the Content for youtube video"):
              ## Validate all the inputs
              if not groq_api_key.strip() or not generic_url.strip():
                  st.error("Please provide the information to get started")
              elif not validators.url(generic_url):
                  st.error("Please enter a valid youtube Url.")

              else:
                  try:
                      with st.spinner("Waiting..."):
                          ## loading the website or yt video data
                          
                          loader=YoutubeLoader.from_youtube_url(generic_url,add_video_info=True)
                          docs=loader.load()

                          ## Chain For Summarization
                          chain=load_summarize_chain(llm,chain_type="stuff",prompt=prompt)
                          output_summary=chain.run(docs)

                          st.success(output_summary)
                  except Exception as e:
                      st.exception(f"Exception:{e}")
                              
       
          `}
      />
      <p>
        <p>
          A Streamlit web application designed to summarize YouTube videos using
          AI. The application leverages the LangChain library and the Groq API
          to generate concise summaries of video content based on transcripts
          obtained from YouTube.First, we import the necessary libraries,
          including validators for URL validation, streamlit for the web
          interface, and various components from the LangChain library for
          handling prompts and summarization. We also import os to manage
          environment variables.
          <br />
          The Streamlit app is configured with a title and a brief description.
          The user is prompted to input a YouTube URL, which is then validated.
          If the URL is valid, the app proceeds to load the video data using the
          YoutubeLoader from LangChain. This loader extracts the transcript and
          other relevant information from the provided YouTube URL. We then
          define a prompt template for the summarization task. This template
          instructs the AI to provide a summary of the content in 300 words. The
          ChatGroq model is initialized with the Groq API key and the specified
          model, Gemma-7b-It.
          <br />
          When the user clicks the “Summarize the Content for youtube video”
          button, the app validates the inputs. If the inputs are valid, it
          loads the video data and runs the summarization chain using the
          defined prompt. The summary is then displayed to the user.
          <br />
        </p>
        
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
      {/* <div className="text-center">
        <p>Deployed App</p>
        <img
          src={img01}
          alt="result1"
          style={{ height: "400px", width: "600px" }}
        />
      </div> */}
    </div>
  );
};

export default Content025;
