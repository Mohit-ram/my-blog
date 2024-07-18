import Code from "../../components/Code/Code.jsx";
import img01 from "./img01.png";

const Content018 = () => {
  return (
    <div className="page container mt-5 mx-auto w-75 px-5 ">
      <h1 className="text-center">RAG Framework: Vectore Store DB</h1>
      <div className="text-center"></div>
      <p>
        This project focuses on the implementation and utilization of advanced
        vector stores for document similarity searches. The primary objective is
        to explore and demonstrate the capabilities of FAISS (Facebook AI
        Similarity Search) and Chroma vector stores in handling and processing
        large text datasets. By leveraging these technologies, we aim to enhance
        the efficiency and accuracy of document retrieval systems.
      </p>
      <h4>Data Loading</h4>
      <Code
        code={`
          #Step1 Data loading and splitting
          # Import the necessary classes from the langchain_community and langchain_text_splitters modules
          from langchain_community.document_loaders import TextLoader
          from langchain_text_splitters import CharacterTextSplitter
          # Instantiate the TextLoader with the path to the text file
          loader = TextLoader('sample_story.txt')
          # Load the contents of the text file into the 'documents' variable
          documents = loader.load()
          # Create an instance of CharacterTextSplitter with specified chunk size and overlap
          text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=30)
          # Split the loaded documents into smaller chunks and store them in the 'docs' variable
          docs = text_splitter.split_documents(documents)          
          `}
      />
      <div className=" text-center">
        <p>Input text file</p>
        <img
          src={img01}
          alt="result1"
          style={{ height: "400px", width: "600px" }}
        />
      </div>
      <p>
        The above code demonstrates the process of loading and splitting text
        documents using the langchain_community and langchain_text_splitters
        libraries. Initially, the TextLoader class from the
        langchain_community.document_loaders module is imported to facilitate
        the loading of text files. The CharacterTextSplitter class from the
        langchain_text_splitters module is also imported to handle the splitting
        of text into smaller chunks.
        <br />
        The TextLoader is instantiated with the file path 'sample_story.txt',
        and the load method is called to read the contents of the file into the
        documents variable. Subsequently, an instance of CharacterTextSplitter
        is created with specified parameters: chunk_size=100 and
        chunk_overlap=30. These parameters define the size of each text chunk
        and the overlap between consecutive chunks, respectively. Finally, the
        split_documents method is invoked on the text_splitter instance, passing
        the loaded documents to produce the split text chunks stored in the docs
        variable.
        <br />
        <br />
      </p>

      <h4>FAISS DB</h4>
      <Code
        code={`
          #FAISS Vector store
          from langchain_community.embeddings import OllamaEmbeddings
          # Instantiate the OllamaEmbeddings with the specified model
          embeddings = OllamaEmbeddings(model="phi3")

          # Define the query string for the similarity search
          query = "cave, where a magnificent chest lay waiting. As she opened the "
          docs_and_score = db.similarity_search_with_score(query)
          # Save the FAISS vector store locally with the specified filename
          db.save_local("faiss_db")
          # Load the FAISS vector store from the local file with the specified embeddings
          new_db = FAISS.load_local("faiss_Vdb", embeddings, allow_dangerous_deserialization=True)
          # Perform a similarity search using the query on the new database object
          docs = new_db.similarity_search(query)          
          `}
      />
      
      <p>
        The provided code snippet demonstrates the process of creating and
        utilizing a FAISS (Facebook AI Similarity Search) vector store for
        document similarity searches. Initially, the OllamaEmbeddings class from
        the langchain_community.embeddings module is imported and instantiated
        with the model "phi3". This model is used to generate embeddings for the
        documents. A query string is defined, and the
        similarity_search_with_score method is called on the db object to find
        documents similar to the query, along with their similarity scores. The
        results are stored in the docs_and_score variable. The db object is then
        saved locally using the save_local method with the filename "faiss_db".
        Subsequently, a new FAISS vector store is loaded from the local file
        using the FAISS.load_local method.
        <br />
        <br />
      </p>
      <h4>Chroma DB</h4>
      <Code
        code={`
          from langchain_chroma import Chroma
          # Create a Chroma vector store from docs using the specified embeddings
          chroma_vdb = Chroma.from_documents(documents=docs, embedding=embeddings)

          # Save the entire set of documents to a Chroma vector store on disk in the specified directory
          vectordb = Chroma.from_documents(documents=docs, embedding=embeddings, persist_directory="./chroma_db")
          # Load the Chroma vector store from the disk using the specified directory and embeddings
          db2 = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
          # Perform a similarity search on the loaded vector store using the specified query
          docs = db2.similarity_search(query)         
          `}
      />
      <p>
        The Chroma.from_documents method is used to create a vector store from
        the first two documents in the docs list, using the specified
        embeddings. This vector store is stored in the chroma_vdb variable.
        Next, the entire set of documents is used to create another Chroma
        vector store, which is saved to disk in the specified directory
        ./chroma_db using the persist_directory parameter. This vector store is
        stored in the vectordb variable. Subsequently, the Chroma vector store
        is loaded from the disk using the Chroma constructor with the
        persist_directory parameter set to ./chroma_db and the
        embedding_function parameter set to the previously defined embeddings.
        This loaded vector store is stored in the db2 variable.
        <br />
        <br />
      </p>
      <h5>Similarity search Output.</h5>
      <Code
        code={`
          [(Document(metadata={'source': 'sample_story.txt'}, page_content='Upon reaching the island, she discovered a cave entrance hidden behind a waterfall. Inside the cave, she found a series of intricate puzzles and traps designed to protect the treasure. Using her wit and determination, Elara solved each puzzle, narrowly escaping the traps.'),
            3823.5085),
          (Document(metadata={'source': 'sample_story.txt'}, page_content='Elara realized that the true treasure was not the riches, but the journey and the knowledge she had gained. She decided to return to her village and share her newfound wisdom with her people. The journey back was filled with new challenges, but Elara faced them with confidence and courage.'),
            4018.107),
          (Document(metadata={'source': 'sample_story.txt'}, page_content='When she finally returned home, the villagers were amazed by her tales and the knowledge she brought back. Elara’s adventure inspired others to seek their own paths and explore the world beyond the village. She became a beloved figure, known for her bravery and wisdom.'),
            4141.683),
          (Document(metadata={'source': 'sample_story.txt'}, page_content='Finally, she reached the heart of the cave, where a magnificent chest lay waiting. As she opened the chest, a blinding light filled the cave, revealing a treasure beyond her wildest dreams. But more than gold and jewels, the chest contained ancient scrolls filled with knowledge and wisdom.'),
            4170.613)]
          
          `}
      />
    </div>
  );
};

export default Content018;
