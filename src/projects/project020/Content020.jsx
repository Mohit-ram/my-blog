import Code from "../../components/Code/Code.jsx";


const Content020 = () => {
  return (
    <div className="page container mt-5 mx-auto w-75 px-5 ">
      <h1 className="text-center">A Simple Q&A Model</h1>
      <div className="text-center">
       
      </div>
      <p>
        In This project we develop a simple Q&A AI model using lancgchain models
        and promt techniques. The model used is free source Gemma model throud
        GROQ interface. It outputs what ever question you ask. In the page I
        have also include code to access differnet AI model for project and also
        include code for different prompt techniques.
      </p>
      <h4>Basic Q&A LLM model</h4>
      <Code
        code={`
          #simple Query app using Gemma model
          import os
          from dotenv import load_dotenv

          # Load all environment variables from the .env file
          load_dotenv()

          # Initialize the GroQ model with the API key
          groq_api_key = os.getenv("GROQ_API_KEY")
          from langchain_groq import ChatGroq
          model = ChatGroq(model="Gemma-7b-It", groq_api_key=groq_api_key)

          # Invoke the model to tell a funny joke about fruits
          model.invoke("tell me a funny joke about fruits")

          # Import necessary classes for prompting the model
          from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
          from langchain_core.messages import SystemMessage, HumanMessage

          # Create a prompt template to ask about a topic in 10 words
          prompt_template = PromptTemplate.from_template(
              "Tell me little about {topic}" + " restrict to 10 words"
          )
          prompt = prompt_template.format(topic='AI')

          # Create a chat prompt template for a simulated conversation
          chat_template = ChatPromptTemplate.from_messages(
              [
                  ("system", "You are a helpful AI bot. Your name is {name}."),
                  HumanMessage(content="Hello, how are you doing?"),
                  SystemMessage(content="I'm doing well, thanks!"),
                  ("human", "{user_input}"),
              ]
          )
          chat_prompt = chat_template.format_messages(
              name="Bob",
              user_input="What is your name?"
          )

          # Invoke the model with the simple prompt
          response1 = model.invoke(prompt)

          # Invoke the model with the chat prompt
          response2 = model.invoke(chat_prompt)

          
          `}
      />
      <p>
        WE begin by loading environment variables from a .env file using the
        dotenv library, which is essential for securely managing API keys and
        other sensitive information. The first section initializes the GroQ
        model, an open-source language model. The API key for GroQ is retrieved
        from the environment variables, and the ChatGroq class from the
        langchain_groq module is used to initialize the model with the specified
        parameters. The model is then invoked to tell a funny joke about fruits.
        <br />
        Next, the code sets up prompting for the model. It imports necessary
        classes from the langchain_core module and creates a PromptTemplate to
        generate a prompt that asks the model to provide a brief description of
        a given topic, restricted to 10 words. This template is then formatted
        with the topic ‘AI’.
        <br />
        We also creates a ChatPromptTemplate to simulate a chat interaction. It
        includes system and human messages to guide the conversation. The chat
        prompt is formatted with the name “Bob” and a user input asking for the
        bot’s name.
        <br />
        <br />
      </p>
      <h4>Various models in Langchain</h4>
      <Code
        code={`
          import os
          from dotenv import load_dotenv

          # Load all environment variables from the .env file
          load_dotenv()

          # GroQ model - open source
          groq_api_key = os.getenv("GROQ_API_KEY")
          from langchain_groq import ChatGroq
          model = ChatGroq(model="Gemma2-9b-It", groq_api_key=groq_api_key)

          # Import various LLMs from LangChain
          from langchain.llm import OpenAI
          from langchain.llm import VertexAI
          from langchain_community.llms import Ollama
          # Initialize the Ollama model with "llama3"
          llm = Ollama(model="llama3")
          llm.invoke("Tell me a joke")

          # Import chat models with appropriate API keys saved in .env file
          from langchain_openai import ChatOpenAI
          from langchain_google_vertexai import ChatVertexAI
          from langchain_mistralai import ChatMistralAI

          # Initialize chat models with specified models and API keys
          chat_openai = ChatOpenAI(model="gpt-4")
          chat_vertexai = ChatVertexAI(model="gemini-pro")
          chat_mistralai = ChatMistralAI(model="mistral-large-latest", api_key="...")

          `}
      />
      <h4>Prompt Templates in Langchain</h4>
      <Code
        code={`
          #Prompts
          # Use PromptTemplate to create a template for a string prompt.
          from langchain_core.prompts import PromptTemplate
          prompt_template = PromptTemplate.from_template(
              "Tell me a {adjective} joke about {content}."
          )
          prompt_template.format(adjective="funny", content="chickens")
          #output: 'Tell me a funny joke about chickens.'

          #chat Prompt template
          from langchain_core.prompts import ChatPromptTemplate
          chat_template = ChatPromptTemplate.from_messages(
              [
                  ("system", "You are a helpful AI bot. Your name is {name}."),
                  ("human", "Hello, how are you doing?"),
                  ("ai", "I'm doing well, thanks!"),
                  ("human", "{user_input}"),
              ]
          )
          messages = chat_template.format_messages(
                                              name="Bob",
                                              user_input="What is your name?")
          #Chat Promt Template using message inputs
          from langchain_core.messages import SystemMessage
          from langchain_core.prompts import HumanMessagePromptTemplate
          chat_template = ChatPromptTemplate.from_messages(
              [
                  SystemMessagePromptTemplate.from_template("You are a helpful AI bot. Your name is {name}"),
                  HumanMessage(content="Hello, how are you doing?"),
                  ("ai", "I am doing well, thanks!"),
                  HumanMessagePromptTemplate.from_template("{user_input}"),
              ]
          )

          #Message Placeholders
          from langchain_core.prompts import   ChatPromptTemplate, HumanMessagePromptTemplate,MessagesPlaceholder,

          human_prompt = "Summarize our conversation so far in {word_count} words."
          human_message_template = HumanMessagePromptTemplate.from_template(human_prompt)

          chat_prompt = ChatPromptTemplate.from_messages(
              [MessagesPlaceholder(variable_name="conversation"), human_message_template]
          )
          `}
      />
      <p>
        <br />
        <br />
      </p>
    </div>
  );
};

export default Content020;
