import thb from "/src/projects/project-thumbnails.js";


const projects = [
  {
    Id: 1,
    number: "001",
    title: "The well know beginner Iris Species detection",
    info: "Accurately predicting the species of Iris flower based on flower parameters using machine learning algorithms. ",
    subInfo:
      " Classification, Scikit-learn, Hypertuning Random Forest, Cross-validation, GradBoost, RadomisedCVSearch",
    imgPath: thb[1],
    category: "cat-a",
    dataSource: "link",
  },
  {
    Id: 2,
    number: "002",
    title: "Determining the Age of Abalones",
    info: "Appliying of machine learning techniques to the abalone dataset to predict the age category of abalones based on physical measurements.",
    subInfo:
      "Classification, Column Transformer, Confusion matrix, F1 score, support vector machines",
    imgPath: thb[2],
    category: "cat-a",
    dataSource: "https://archive.ics.uci.edu/dataset/1/abalone",
  },
  {
    Id: 3,
    number: "003",
    title: "Handwritten Digit Recognition",
    info: "Handwritten digit recognition, exploring deep learning techniques from data preprocessing to model architecture and regularization",
    subInfo:
      "Multi-class Image classification, CNN, overfitting, Regularisation PerformaceCurves",
    imgPath: thb[3],
    category: "cat-a",
    dataSource: "https://www.kaggle.com/competitions/digit-recognizer",
  },
  {
    Id: 4,
    number: "004",
    title: "Human or Horse",
    info: "Appliying transfer learning techniques to the classify humans or horses. ",
    subInfo: "Inceptionv3, Transfer Learning, layer weights, prefetching",
    imgPath: thb[4],
    category: "cat-b",
    dataSource: "https://laurencemoroney.com/datasets.html",
  },
  {
    Id: 5,
    number: "005",
    title: "Cat or Dog a transfer learning approach",
    info: "A Inceptionv3 model based classification of cats or dogs in an image. ",
    subInfo:
      "Binary Calssification, Transfer Learning, tf datasets, Autotune, ImagesFromDirectories.",
    imgPath: thb[5],
    category: "cat-b",
    dataSource: "https://www.kaggle.com/c/dogs-vs-cats",
  },
  {
    Id: 6,
    number: "006",
    title: "Object tracking with yolo",
    Info: "Object detection and tracking objects throughout frames only using YOLO",
    subInfo: "ObjectTrackin ComputerVision OpenCV Yolov8 ",
    imgPath: thb[6],
    category: "cat-c",
    dataSource: "pixabay",
  },
  {
    Id: 7,
    number: "007",
    title: "Wildlife Detection with YOLOv8 and Ultralytics",
    info: "Object detection using YOLOv8 and the Ultralytics datasets. Fine-tune the model on an “african-wildlife” dataset and perform inference on sample images.",
    subInfo:
      "Obeject Detection, Google Colab, Yolov8, Ultralytics Dataset, Yolo metrics, mAP50.",
    imgPath: thb[7],
    category: "cat-c",
    dataSource: "https://docs.ultralytics.com/",
  },
  {
    Id: 8,
    number: "008",
    title: "Object segmentation with YOLO",
    info: "YOLO object instance segmentaion to find packge boxes in an image",
    subInfo: "Object Segmentation, Instance segmentation",
    imgPath: thb[8],
    category: "cat-c",
    dataSource: "link",
  },
  {
    Id: 9,
    number: "009",
    title: "A quick start guide to postgreSQL",
    info: "Quick and all you need to start your quieries in postgreSQL using pgAdmin 4",
    subInfo: "Installation, Create DB, SQL Datatypes, Table, Insert",
    imgPath: thb[9],
    category: "cat-a",
    dataSource: "(https://www.postgresql.org/)",
  },
  {
    Id: 10,
    number: "010",
    title: "SQL Basic queries and data analysis",
    info: "Most used basic sql quries to perfom data analysis on custom created dataset",
    subInfo: "SELECT, WHERE, JOINS, GROUPING, FILTERING, AGGREGATION",
    imgPath: thb[10],
    category: "cat-a",
    dataSource: "",
  },
  {
    Id: 11,
    number: "011",
    title: "SQL advanced queries",
    info: "mUltiple advanced queries to  thoroughly explore and analyse data on a custom created dataset",
    subInfo: "SUBQUERIES, UNIONS, WindowFunctions, OVER, RANK",
    imgPath: thb[11],
    category: "cat-b",
    dataSource: "",
  },
  {
    Id: 12,
    number: "012",
    title: "Quick visualisation using pandas",
    info: "Basic simple pandas functions for quickly visualise directly from dataframes. ",
    subInfo: "Hist plot, scatterPlot, BarPlot, KDEPlot, BoxPlot",
    imgPath: thb[12],
    category: "cat-a",
    dataSource: "link",
  },
  {
    Id: 13,
    number: "013",
    title: "Starting guide to Numpy",
    info: "All the useful numpy functions, concepts, operations in one guide",
    subInfo: "ArrayCreation, MathOperations, nD-arrays, Slicing, Stacking",
    imgPath: thb[13],
    category: "cat-a",
    dataSource: "link",
  },
  {
    Id: 14,
    number: "014",
    title: "Real time Iris Species classification with streamlit interface",
    info: "Using stremlit interface to get input features and classify iris species using Random Forest",
    subInfo: "Streamlit, Random Forest, classification",
    imgPath: thb[14],
    category: "cat-b",
    dataSource: "link",
  },
  
  {
    Id: 16,
    number: "016",
    title: "NLP: Sentiment analysis: Sarcasm or not",
    info: "Sarcasm detection model using deep learning techniques, trained on a dataset of headlines. Streamlit web application for real-time predictions on whether the text is sarcastic.",
    subInfo: "LSTM, Tokenisatoin, streamlit, sequence data",
    imgPath: thb[16],
    category: "cat-b",
    dataSource: "link",
  },
  {
    Id: 17,
    number: "017",
    title: "LangChain: RAG Framework Document embedding",
    info: "Retrieval-Agumented Generation Framework in Langchain, Data ingestion, Data chunking, Data embedding",
    subInfo: "Langchain, Ollama, OpenAI, HuggingFace, Embeddings, Text Loaders, Text Splitters ",
    imgPath: thb[17],
    category: "cat-d",
    dataSource: "link",
  },
  {
    Id: 18,
    number: "018",
    title: "LangChain: RAG Framework Vector Stores",
    info: "Creating vectore store DB for custom document using FAIS and Chroma DB",
    subInfo: "Vector Stores, FIASS, ChromaDB, Ollama Embeddings",
    imgPath: thb[18],
    category: "cat-d",
    dataSource: "link",
  },
  
  {
    Id: 20,
    number: "020",
    title: "A Basic Q&A LLM model using LangChain",
    info: "A simple Q&A AI app using Langchain chatmodels and prompt techniques",
    subInfo: "GROQ, LLM, Langchain Components: Chat Models + Prompt, OpenAI, Ollama",
    imgPath: thb[20],
    category: "cat-d",
    dataSource: "link",
  },
  {
    Id: 21,
    number: "021",
    title: "Langchain: Retain Chat Message history",
    info: "Project demonstrates various chat history retaining techniques in Langchain, that can be later used in chatBots and Agents",
    subInfo: "LangChain, Chains, Prompts, RunnableWithMessageHistory",
    imgPath: thb[21],
    category: "cat-d",
    dataSource: "link",
  },
  
  {Id: 22,
    number: "022",
    title: "Langchain: Deployed World Wars Q&A AI, end-to-end",
    info: "End to End AI app deployment using langchain, Groq, Huggingface, streamlit",
    subInfo: "Langchain, streamlit, HuggingFace, GroQ, Llama",
    imgPath: thb[22],
    category: "cat-d",
    dataSource: "link",
},
{
  Id: 19,
  number: "019",
  title: "NLP: Text Classification",
  info: "Project demonstrates the application of deep learning techniques to NLP tasks: text classification model.",
  subInfo: "Text Classification, Tokenisers, BBC news dataset, Embedding Layers",
  imgPath: thb[19],
  category: "cat-b",
  dataSource: "link",
},
{Id: 23,
  number: "023",
  title: "Contextual Q&A Chat with AI, End-to-End App",
  info: "A AI app that answers based on provided context via pdfs and with chat history.",
  subInfo: "Langchain, History Retreivers,streamlit, HuggingFace, GroQ, Llama",
  imgPath: thb[23],
  category: "cat-d",
  dataSource: "https://contextualqnaai-k4kibvvy8vspmxwbsx9vfv.streamlit.app/)",
},
{Id: 24,
  number: "024",
  title: "WebPage Summary AI app",
  info: "Summarizing valid web page and deploying into cloud , using Langchain, Gemma, streamlit. ",
  subInfo: "End-To-End, Summarize Chain, Unstructure Loader, Groq ",
  imgPath: thb[24],
  category: "cat-d",
  dataSource: "https://texsummarylc-zqmbbyxqndscfy3xr7nzyp.streamlit.app/"   
  ,
},
{
  Id: 15,
  number: "015",
  title: "NLP: End-to-End Sentiment prediction using RNN",
  info: "Build and deploy a sentiment analysis model using the IMDB movie reviews dataset, classifying reviews as positive or negative with a Recurrent Neural Network (RNN).",
  subInfo: "End-to-End, Deploy, RNN, Embedding vector, PadSequences, Streamlit",
  imgPath: thb[15],
  category: "cat-b",
  dataSource: "link",
},
{Id: 25,
  number: "025",
  title: "Youtube Video Summary AI app",
  info: "Summarizing a valid youtube and deploying into cloud , using Langchain, Gemma, streamlit. ",
  subInfo: "End-To-End, Summarize Chain, YT Loader, Groq",
  imgPath: thb[25],
  category: "cat-d",
  dataSource: "link",
}
  
];

function onlyCatA() {
  const catA_projects = projects.filter(function (project) {
    if (project.category == "cat-a") {
      return project;
    }
  });
  return catA_projects;
}

function onlyCatB() {
  const catB_projects = projects.filter(function (project) {
    if (project.category == "cat-b") {
      return project;
    }
  });
  return catB_projects;
}
function onlyCatC() {
  const catC_projects = projects.filter(function (project) {
    if (project.category == "cat-c") {
      return project;
    }
  });
  return catC_projects;
}

function onlyCatD() {
  const catD_projects = projects.filter(function (project) {
    if (project.category == "cat-d") {
      return project;
    }
  });
  return catD_projects;
}

export default projects.reverse();
export { onlyCatA, onlyCatB, onlyCatC, onlyCatD };
