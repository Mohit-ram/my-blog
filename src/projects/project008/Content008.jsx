import Code from "../../components/Code/Code.jsx";
import mainImg from "./train_batch0.jpg";

import img02 from "./BoxPR_curve.png";
import img03 from "./confusion_matrix.png";
import img04 from "./train_batch2400.jpg";
import img05 from "./val_batch0_pred.jpg";
import img06 from "./test_img.jpg";
import img07 from "./prect_img.jpg";

const Content008 = () => {
  return (
    <div className="page container mt-5 mx-auto w-75 px-5 ">
      <h1 className="text-center">Package Segementation with YOLO</h1>
      <div className="text-center">
        <img src={mainImg} className="h-50 w-50"></img>
      </div>
      <p>
        In this project, we leverage the YOLO (You Only Look Once) architecture,
        specifically tailored for segmenting objects within images. The project
        model initialization, training, and making predictions on an image.
        We’ll explore how pre-trained models are used and fine-tuned for
        different datasets, enhancing efficiency and generalization.
      </p>

      <h5>
        !yolo predict model="yolov8s-seg.pt"
        source="https://ultralytics.com/images/bus.jpg"!
      </h5>
      <p>
        The model being used is YOLOv8s-seg, which is specifically designed for
        segmentation tasks. The -seg suffix indicates that it’s a segmentation
        model.The !yolo predict command is used to make predictions with the
        specified model.The source parameter specifies the input image URL:
        "https://ultralytics.com/images/bus.jpg". we use yolo to segment
        different classes in an given image.
        <br />
        <br />
      </p>

      <Code
        code={`
          # Initialize YOLO model for instance segmentation
          from ultralytics import YOLO
          model = YOLO("yolov8n-seg.pt")

          # Train the model using the specified configuration
          results = model.train(data="package-seg.yaml", epochs=30, imgsz=640)

          # Load a pre-trained model (if available)
          loaded_model = YOLO("/contnet/runs/segment/weights/bes.pt")

          # Make predictions on an image
          image_path = "path/to/your/image.jpg"
          result = model.predict(image_path, save=True)  # Save segmentation results
                    `}
      />
      <p>
        Model Initialization and Training: We start by importing the necessary
        libraries. The ultralytics library provides tools for working with YOLO
        models. Next, we create an instance of the YOLO model using the
        pre-trained weights file "yolov8n-seg.pt". This model is specifically
        designed for instance segmentation tasks. We then train the model using
        the "package-seg.yaml" configuration file. The training process runs for
        30 epochs with an input image size of 640x640 pixels. The results
        variable stores information about the training process, including loss
        values, accuracy metrics, and other relevant data.
        <br />
        Loading a Pre-Trained Model: We load another YOLO model from the file
        "/contnet/runs/segment/weights/bes.pt". This model might have been
        trained previously or obtained from another source. The loaded_model
        variable holds this pre-trained model.
        <br />
        Prediction on an Image: Finally,we make predictions using the
        model.predict() method. We provide the path to an image (specified as
        "img-path"), and the model generates instance segmentation results. The
        save=True argument indicates that the predictions should be saved (e.g.,
        as masks or contours) for further analysis or visualization.
        <br />
        <br />
      </p>
      <h4>Results</h4>
      <div className="d-block text-center">
        <img
          src={img02}
          alt="result1"
          style={{ height: "300px", width: "300px" }}
        />
        <img
          src={img03}
          alt="result2"
          style={{ height: "300px", width: "300px" }}
        />
        <img
          src={img04}
          alt="result3"
          style={{ height: "300px", width: "300px" }}
        />
        <img
          src={img05}
          alt="result4"
          style={{ height: "300px", width: "300px" }}
        />
      </div>
      <h4> Yolo Inference</h4>
      <div className="d-block text-center">
        <img
          src={img06}
          alt="result1"
          style={{ height: "300px", width: "400px" }}
        />
        <img
          src={img07}
          alt="result2"
          style={{ height: "300px", width: "400px" }}
        />
      </div>
      <p>
        image 1/1
        /content/datasets/package-seg/test/images/5547_zl20230718_003_png_jpg.rf.4969ed3a70f791acf8bfa3dfbd7aaa95.jpg:
        640x640 6 packages, 25.6ms Speed: 1.8ms preprocess, 25.6ms inference,
        6.2ms postprocess per image at shape (1, 3, 640, 640) Results saved to
        runs/segment/train22
        <br />
        <br />
      </p>
    </div>
  );
};

export default Content008;
