import Code from "../../components/Code/Code.jsx";


const Content017 = () => {
  return (
    <div className="page container mt-5 mx-auto w-75 px-5 ">
      <h1 className="text-center">RAG Framewowk: Document Embedding</h1>
      <div className="text-center">
        
      </div>
      <p>
        Retrieval-Augmented Generation (RAG) is an advanced AI framework
        designed to enhance the performance of large language models (LLMs) by
        integrating them with external data retrieval mechanisms. This approach
        addresses some of the inherent limitations of LLMs, such as the tendency
        to generate outdated or inaccurate information, by grounding their
        responses in up-to-date and authoritative sources.
        <br />
        <h5> How RAG works </h5>
        Query Processing: The system receives a query from the user and
        processes it to understand the context and information needs.
        <br />
        Information Retrieval: The system searches for relevant information from
        external sources, such as databases, documents, or the web. This step
        involves generating vector embeddings of the query and performing
        similarity searches to find the most relevant data2.
        <br />
        Augmentation of the Query: The retrieved information is used to augment
        the original query, providing additional context and details that the
        LLM can use to generate a more accurate response1.
        <br />
        Response Generation: The augmented query is fed into the LLM, which
        generates a response based on both its internal knowledge and the
        retrieved external information2. Delivery of the Response: The system
        delivers the final response to the user, ensuring that it is both
        accurate and contextually relevant1.
        <br />
      </p>
      <h4>Implementation of RAG with Langchain</h4>
      <h5>Stage1: Data Ingestion </h5>
      <p>
        Data ingestion in a Retrieval-Augmented Generation (RAG) framework
        involves several key steps to ensure that the external data is
        effectively integrated and utilized by the language model.
        <br />
        <br />
      </p>
      <Code
        code={`
          #Stage 1 :Data Ingestion
          # Importing necessary classes from langchain_community.document_loaders
          from langchain_community.document_loaders import TextLoader, PyPDFLoader, WebBaseLoader, WikipediaLoader

          # Loading a text file
          # Create a TextLoader instance for the file 'speech.txt'
          loader = TextLoader('sample.txt')
          text_documents = loader.load()

          # Reading a PDF file
          # Create a PyPDFLoader instance for the file 'attention.pdf'
          loader = PyPDFLoader('sample.pdf')
          docs = loader.load()

          # Reading from a website
          # Import BeautifulSoup for HTML parsing
          import bs4
          # Create a WebBaseLoader instance for the given URL
          loader = WebBaseLoader(
              web_paths=(url,),
              bs_kwargs=dict(parse_only=bs4.SoupStrainer(class_=("post-title", "post-content", "post-header")))
          )
          loader.load()
          # Reading from a Wikipedia article
          # Create a WikipediaLoader instance to query "any article_name" and load a maximum of 2 documents
          docs = WikipediaLoader(query="article_name", load_max_docs=2).load()
          `}
      />
      <h5>Stage2 :Data chunking</h5>
      <p>
        Data chunking is a technique used to divide large datasets or documents
        into smaller, more manageable pieces called chunks. This process is
        essential in various fields, including natural language processing, data
        storage, and data transmission, to ensure efficient processing,
        retrieval, and analysis.
        <br />
        <br />
      </p>
      <Code
        code={`
          #Data Chunking
          # Importing necessary classes from langchain_text_splitters
          from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter, HTMLHeaderTextSplitter, RecursiveJsonSplitter

          # Recursive Character Splitter
          # Create a RecursiveCharacterTextSplitter instance with chunk size of 500 and overlap of 50
          text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
          # Split documents into chunks
          final_documents = text_splitter.split_documents(docs)  # for input in documents
          text = text_splitter.create_documents("list of texts")  # for input in text

          # Character Splitter
          # Create a CharacterTextSplitter instance with separator as double newline, chunk size of 100, and overlap of 20
          text_splitter = CharacterTextSplitter(separator="\n\n", chunk_size=100, chunk_overlap=20)
          text_splitter.split_documents(docs)

          # HTML Header Splitter
          # Define headers to split on
          headers_to_split_on = [
              ("h1", "Header 1"),
              ("h2", "Header 2"),
              ("h3", "Header 3")
          ]
          # Create an HTMLHeaderTextSplitter instance with specified headers
          html_splitter = HTMLHeaderTextSplitter(headers_to_split_on)
          # Split HTML content into chunks
          html_header_splits = html_splitter.split_text(html_string)

          # JSON Splitter
          # Create a RecursiveJsonSplitter instance with max chunk size of 300
          json_splitter = RecursiveJsonSplitter(max_chunk_size=300)
          # Split JSON data into chunks
          json_chunks = json_splitter.split_json(json_data)          
          `}
      />
      <h5>Stage3: Data Embedding</h5>
      <p>
        Embeddings are numerical representations of text that capture semantic
        meaning, making them useful for various natural language processing
        tasks such as similarity searches and information retrieval.
        <br />
        <br />
      </p>
      <Code
        code={`
          #Document/text Embedding
          # Importing necessary modules
          from dotenv import load_dotenv
          # Load all the environment variables from the .env file
          load_dotenv()

          # OpenAI embeddings - paid source
          # Set the OpenAI API key from environment variables
          os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
          from langchain_openai import OpenAIEmbeddings
          # Create an instance of OpenAIEmbeddings with specified model and dimension
          embeddings = OpenAIEmbeddings(model="text-embedding-3-large", dimension=1024)
          text = "This is a tutorial on OPENAI embedding"
          # Generate an embedding for the text query
          query_result = embeddings.embed_query(text)

          # Ollama embeddings - open source
          from langchain_community.embeddings import OllamaEmbeddings
          # Create an instance of OllamaEmbeddings for model of your choics model name from ollama web
          embeddings = OllamaEmbeddings(model="model_name")
          # Generate embeddings for a list of documents
          query_result = embeddings.embed_documents("[list of documents]")
          # Generate an embedding for a single text query
          query_result = embeddings.embed_query("single text query")

          # Hugging Face embeddings - open source
          # Set the Hugging Face API token from environment variables
          os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")
          from langchain_huggingface import HuggingFaceEmbeddings
          # Create an instance of HuggingFaceEmbeddings with the specified model
          embeddings = HuggingFaceEmbeddings(model_name="any_model_name")
          # Generate an embedding for a single text query
          query_result = embeddings.embed_query("any single text")

          `}
      />
    </div>
  );
};

export default Content017;
