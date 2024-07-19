import Code from "../../components/Code/Code.jsx";

const Content021 = () => {
  return (
    <div className="page container mt-5 mx-auto w-75 px-5 ">
      <h1 className="text-center">LLM Chat History</h1>
      <div className="text-center"></div>
      <p>
        This is an Intermediate project and, in this page first we see how
        general chains fail to retreive chat message history. Then we use
        RunnableWithMeassageHistory classes to retain history in two case, one
        without prompt and later with prompt_template.
      </p>
      <h5>common chain Runnable</h5>
      <Code
        code={`
          #History Check
          #Load model
          import os
          from dotenv import load_dotenv
          load_dotenv() ## aloading all the environment variable

          groq_api_key=os.getenv("GROQ_API_KEY")
          from langchain_groq import ChatGroq
          model=ChatGroq(model="Gemma2-9b-It",groq_api_key=groq_api_key)

          #check message history
          from langchain_core.messages import HumanMessage
          out1= model.invoke([HumanMessage(content="Hi , My name is Moiht, I am an AI engineer")])
          out2= model.invoke([HumanMessage(content="what is my name?")])
          `}
      />
      <p>
        out1: AIMessage(content="Hello Moiht, it's nice to meet you! )
        <br />
        out2: AIMessage(content="As an AI, I have no memory of past
        conversations
        <br />
        AS We can observe the LLM model does not have any memory of previous
        query/messages and can't respond appropreatly.
      </p>
      <h4>RunnableWithMessageHistory</h4>
      <Code
        code={`
          #Step3#To retain message history
          from langchain_community.chat_message_histories import ChatMessageHistory
          from langchain_core.chat_history import BaseChatMessageHistory
          from langchain_core.runnables.history import RunnableWithMessageHistory

          store={}

          def get_session_history(session_id:str)->BaseChatMessageHistory:
              if session_id not in store:
                  store[session_id]=ChatMessageHistory()
              return store[session_id]

          with_message_history=RunnableWithMessageHistory(model,get_session_history)

          #config to chat1
          config_1 = {"configurable":{'session_id':'chat1'}}
          response1 = with_message_history.invoke(
              [HumanMessage(content="Hi , My name is Moiht, I am an AI engineer")],
              config=config_1
          )
          response2 = with_message_history.invoke(
              [HumanMessage(content="What is my name")],
              config=config_1
          )

          
          `}
      />
      <p>
        response1 AIMessage(content="Hi Moiht, \n\nWelcome! It's great to have
        another AI engineer in the mix.)
        <br />
        response2 AIMessage(content='Your name is Moiht. 😊 \n\nI remember that
        you introduced yourself! Is there anything else I can help you with?\n')
        <br />
        As we can see the model have previous conversations memory from chat1
        session and it can use them to generate reasonable responses.
      </p>
      <h4>RunnableChains With Prompt</h4>
      <Code
        code={`
          #Step4 MessageHistory with prompt template
          from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

          prompt = ChatPromptTemplate.from_messages(
              [
                  (
                      "system",
                      "You are a helpful assistant. Answer all questions to the best of your ability in {language}.",
                  ),
                  MessagesPlaceholder(variable_name="messages"),
              ]
          )
          #LCEL chaining components
          chain = prompt|model
          with_message_history_new=RunnableWithMessageHistory(chain,
                                                              get_session_history,
                                                              input_messages_key="messages")

          config_2 = {"configurable":{'session_id':'chat2'}}
          response1 = with_message_history_new.invoke(
              {'messages':[HumanMessage(content="my name is Mohit")], 'language': 'French'},
              config=config_2
          )
          response2 = with_message_history_new.invoke(
              {'messages':[HumanMessage(content="what is my name")], 'language': 'spanish'},
              config=config_2
          )          
          `}
      />
      <p>
        response1 AIMessage(content="Enchanté, Mohit ! \n\nC'est un plaisir de
        vous connaître.)
        <br />
        response2 AIMessage(content='Tu te llamas Mohit)
        <br />
        In this case we create a simple chain = prompt + model first, and then we invoke
        model with additional parameters like language to generate prompt. The model can
        remember previous convo and able to give responses.
      </p>
    </div>
  );
};

export default Content021;
