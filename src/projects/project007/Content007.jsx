import Code from "../../components/Code/Code.jsx";
import mainImg from "./mainImg007.jpg";
import img01 from "./training_metrics.png";
import img02 from "./PR_curve.png";
import img03 from "./confusion_matrix.png";
import img04 from "./train_batch0.jpg";
import img05 from "./val_batch0_labels.jpg";
import img06 from "./elephants-1900332_1280.jpg";
import img07 from "./elephants-1900332_1280 (1).jpg";

const Content007 = () => {
  return (
    <div className="page container mt-5 mx-auto w-75 px-5 ">
      <h1 className="text-center">African Wildlife detection </h1>
      <div className="text-center">
        <img src={mainImg} className="h-50 w-50"></img>
      </div>
      <p>
        Object detection is a fundamental task in computer vision, enabling us
        to identify and locate objects within an image. In this project, we’ll
        explore how to use YOLOv8 (You Only Look Once version 8) along with the
        Ultralytics library to perform object detection. YOLOv8 is a
        state-of-the-art real-time object detection model known for its speed
        and accuracy.
      </p>
      <h4></h4>
      <Code
        code={`
            #Step1
            !nvidia-smi
            !pip install ultralytics
            from ultralytics import YOLO
            # Load a model
            model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)
            # Train the model
            results = model.train(data="african-wildlife.yaml", epochs=30, imgsz=640)
            # Load a model
            model = YOLO("/content/runs/detect/train/weights/best.pt")  # load a brain-tumor fine-tuned model
            # Inference using the model
            results = model.predict("/content/elephants-1900332_1280.jpg", save=True)
                      
          `}
      />
      <p>
        We begin by importing necessary libraries and modules. Specifically,
        import the YOLO class from the ultralytics package. The YOLO (You Only
        Look Once) algorithm is a popular object detection model that can
        identify and locate multiple objects within an image. Next, we load a
        pre-trained YOLO model using the file "yolov8n.pt". Pre-trained models
        are recommended for training because they have already learned useful
        features from a large dataset. The model object now contains the
        pre-trained YOLO model.
        <br />
        
        Now, we proceeds to train the YOLO model using the specified parameters:
        data="african-wildlife.yaml": This points to a YAML
        (https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/african-wildlife.yaml.)
        file containing information about the dataset used for training. The
        dataset contains labeled images of African wildlife provide by
        ultralytics. epochs=30: The model will be trained for 30 epochs
        (iterations over the entire dataset). imgsz=640: The input image size
        during training is set to 640x640 pixels.
        <br />
        
        Training involves adjusting the model’s weights based on the dataset to
        improve its ability to detect objects accurately. After training, yolo
        result contain various prameters and perfomance metrics to evaluate the
        model. The best model weight are also save as best.pt in
        detect/train/weights. We load the best weights by calling YOLO on it.
        Finally, we perform the inference (object detection) using the loaded
        model. It predicts objects in an image specified by the file path The
        save=True parameter indicates that the results (bounding boxes, class
        labels, confidence scores) should be saved in to an image.
        <br />
        
      </p>

      <div className="text-center ">
        <img className="h-75 w-75" src={img01}></img>
        <p> Training epochs in colab</p>
      </div>
      <h4>Yolo Metrics:</h4>
      <p>
        Box Loss (box_loss):The box loss measures the discrepancy between
        predicted bounding boxes and ground truth bounding boxes. It penalizes
        incorrect localization (i.e., inaccurate bounding box coordinates).
        Lower box loss indicates better localization accuracy.
        <br />
        <br />
        Class Loss (cls_loss):The class loss evaluates the correctness of
        predicted class labels. It penalizes misclassifications (e.g.,
        predicting “car” when the true label is “person”). Lower class loss
        indicates better classification performance.
        <br />
        <br />
        DFL Loss (dfl_loss): DFL (Distribution Focal Loss) is an improvement
        over the standard focal loss. It addresses class imbalance by
        emphasizing hard-to-classify examples. DFL loss combines both
        localization and classification aspects.
        <br />
        <br />
        mAP50 (Mean Average Precision @ IoU 0.5): mAP50 evaluates object
        detection accuracy across different object classes. It computes the
        average precision (AP) at IoU (Intersection over Union) threshold of
        0.5. Higher mAP50 indicates better overall performance.
        <br />
        <br />
        mAP50-90 (Mean Average Precision @ IoU 0.5-0.9): mAP50-90 considers a
        range of IoU thresholds (from 0.5 to 0.9). It provides a more
        comprehensive assessment of detection quality. Higher mAP50-90 reflects
        better performance across various IoU levels.
        <br />
        <br />
        Recall (R): Recall measures the proportion of true positive detections
        out of all actual positive instances. It indicates how well the model
        captures relevant objects. Higher recall means fewer missed detections.
        Precision (P): Precision calculates the proportion of true positive
        detections out of all predicted positive instances. It assesses the
        model’s ability to avoid false positives. Higher precision means fewer
        false alarms.
        <br />
        <br />
      </p>

      <h4> Yolo Results</h4>
      <p>
        Precision-Recall Curve: The precision-recall curve is a graphical
        representation that helps us understand the trade-off between precision
        and recall for different confidence thresholds in object detection.
        Precision: Measures the accuracy of positive predictions. It’s the ratio
        of true positives to the total number of positive predictions (true
        positives + false positives). Recall: Also known as sensitivity or true
        positive rate, it measures the proportion of actual positive instances
        correctly detected by the model. The curve shows how precision and
        recall change as we vary the confidence threshold. A higher threshold
        leads to higher precision but lower recall, while a lower threshold
        increases recall but may reduce precision1.
        <br />
        <br />
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
          image 1/1 /content/elephants-1900332_1280.jpg: 384x640 2 buffalos, 3
          elephants, 102.1ms Speed: 1.9ms preprocess, 102.1ms inference, 2.1ms
          postprocess per image at shape (1, 3, 384, 640) Results saved to
          runs/detect/predict2
        </p>
      </p>
    </div>
  );
};

export default Content007;
