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
    category: "cat-b",
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
    category: "cat-a",
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
    number: "001",
    title: "Sample Project Title 8",
    info: "Quis autem vel eum iure reprehenderit qui in ea voluptate velit esse quam nihil molestiae consequatur.",
    subInfo: "subinfo",
    imgPath: thb[5],
    category: "cat-c",
    dataSource: "link",
  },
  {
    Id: 9,
    number: "001",
    title: "Sample Project Title 9",
    info: "Vel illum qui dolorem eum fugiat quo voluptas nulla pariatur?",
    subInfo: "subinfo",
    imgPath: thb[5],
    category: "cat-a",
    dataSource: "link",
  },
  {
    Id: 10,
    number: "001",
    title: "Sample Project Title 10",
    info: "At vero eos et accusamus et iusto odio dignissimos ducimus qui blanditiis praesentium voluptatum deleniti.",
    subInfo: "subinfo",
    imgPath: thb[5],
    category: "cat-b",
    dataSource: "link",
  },
  {
    Id: 11,
    number: "001",
    title: "Sample Project Title 11",
    info: "Et harum quidem rerum facilis est et expedita distinctio.",
    subInfo: "subinfo",
    imgPath: thb[5],
    category: "cat-c",
    dataSource: "link",
  },
  {
    Id: 12,
    number: "001",
    title: "Sample Project Title 12",
    info: "Nam libero tempore, cum soluta nobis est eligendi optio cumque nihil impedit.",
    subInfo: "subinfo",
    imgPath: thb[5],
    category: "cat-b",
    dataSource: "link",
  },
  {
    Id: 13,
    number: "001",
    title: "Sample Project Title 13",
    info: "Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur.",
    subInfo: "subinfo",
    imgPath: thb[5],
    category: "cat-a",
    dataSource: "link",
  },
  {
    Id: 14,
    number: "001",
    title: "Sample Project Title 14",
    info: "Neque porro quisquam est, qui dolorem ipsum quia dolor sit amet, consectetur, adipisci velit.",
    subInfo: "subinfo",
    imgPath: thb[5],
    category: "cat-b",
    dataSource: "link",
  },
  {
    Id: 15,
    number: "001",
    title: "Sample Project Title 15",
    info: "Quis autem vel eum iure reprehenderit qui in ea voluptate velit esse quam nihil molestiae consequatur.",
    subInfo: "subinfo",
    imgPath: thb[5],
    category: "cat-c",
    dataSource: "link",
  },
  {
    Id: 16,
    number: "001",
    title: "Sample Project Title 16",
    info: "Vel illum qui dolorem eum fugiat quo voluptas nulla pariatur?",
    subInfo: "subinfo",
    imgPath: thb[5],
    category: "cat-a",
    dataSource: "link",
  },
  {
    Id: 17,
    number: "001",
    title: "Sample Project Title 17",
    info: "At vero eos et accusamus et iusto odio dignissimos ducimus qui blanditiis praesentium voluptatum deleniti.",
    subInfo: "subinfo",
    imgPath: thb[5],
    category: "cat-b",
    dataSource: "link",
  },
  {
    Id: 18,
    number: "001",
    title: "Sample Project Title 18",
    info: "Et harum quidem rerum facilis est et expedita distinctio.",
    subInfo: "subinfo",
    imgPath: thb[5],
    category: "cat-c",
    dataSource: "link",
  },
  {
    Id: 19,
    number: "001",
    title: "Sample Project Title 19",
    info: "Nam libero tempore, cum soluta nobis est eligendi optio cumque nihil impedit.",
    subInfo: "subinfo",
    imgPath: thb[5],
    category: "cat-b",
    dataSource: "link",
  },
  {
    Id: 20,
    number: "001",
    title: "Sample Project Title 20",
    info: "Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur.",
    subInfo: "subinfo",
    imgPath: thb[5],
    category: "cat-a",
    dataSource: "link",
  },
  {
    Id: 21,
    number: "001",
    title: "Sample Project Title 21",
    info: "Neque porro quisquam est, qui dolorem ipsum quia dolor sit amet, consectetur, adipisci velit.",
    subInfo: "subinfo",
    imgPath: thb[5],
    category: "cat-b",
    dataSource: "link",
  },
  {
    Id: 22,
    number: "001",
    title: "Sample Project Title 22",
    info: "Quis autem vel eum iure reprehenderit qui in ea voluptate velit esse quam nihil molestiae consequatur.",
    subInfo: "subinfo",
    imgPath: thb[5],
    category: "cat-c",
    dataSource: "link",
  },
  {
    Id: 23,
    number: "001",
    title: "Sample Project Title 23",
    info: "Vel illum qui dolorem eum fugiat quo voluptas nulla pariatur?",
    subInfo: "subinfo",
    imgPath: thb[5],
    category: "cat-a",
    dataSource: "link",
  },
  {
    Id: 24,
    number: "001",
    title: "Sample Project Title 24",
    info: "At vero eos et accusamus et iusto odio dignissimos ducimus qui blanditiis praesentium voluptatum deleniti.",
    subInfo: "subinfo",
    imgPath: thb[5],
    category: "cat-b",
    dataSource: "link",
  },
  {
    Id: 25,
    number: "001",
    title: "Sample Project Title 25",
    info: "Et harum quidem rerum facilis est et expedita distinctio.",
    subInfo: "subinfo",
    imgPath: thb[5],
    category: "cat-c",
    dataSource: "link",
  },
  {
    Id: 26,
    number: "001",
    title: "Sample Project Title 26",
    info: "Nam libero tempore, cum soluta nobis est eligendi optio cumque nihil impedit.",
    subInfo: "subinfo",
    imgPath: thb[5],
    category: "cat-b",
    dataSource: "link",
  },
  {
    Id: 27,
    number: "001",
    title: "Sample Project Title 27",
    info: "Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur.",
    subInfo: "subinfo",
    imgPath: thb[5],
    category: "cat-a",
    dataSource: "link",
  },
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

export default projects;
export { onlyCatA, onlyCatB, onlyCatC, onlyCatD };
