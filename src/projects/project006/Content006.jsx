import Code from "../../components/Code/Code.jsx";
import mainImg from "./mainImg006.png";

const Content006 = () => {
  return (
    <div className="page container mt-5 mx-auto w-75 px-5 ">
      <h1 className="text-center">Obejct Tracking With YOLO</h1>
      <div className="text-center">
        <img src={mainImg} className="h-50 w-50"></img>
      </div>
      <p>
        The purpose of this quick and easy project is to detect different
        objects in a video source and track them in all subsequent frames using
        only yolov8 functionalities.
      </p>
      <h4></h4>
      <Code
        code={`              
          # Import necessary libraries
          from ultralytics import YOLO
          import cv2
          # Load the YOLO model weights if already downloaded
          model = YOLO("../yolov8n.pt")
          # Load the video file
          cap = cv2.VideoCapture("object_tracking_yolo/beach_01.mp4")
          # Initialize loop control variable
          ret = True
          while ret:
              # Read the next frame from the video
              ret, frame = cap.read()
              # Detect and track objects using YOLO
              results = model.track(frame, persist=True)
              frame_ = results[0].plot()
              # Display the tracked frame
              cv2.imshow('video', frame_)
              # Exit loop if 'q' key is pressed
              if cv2.waitKey(0) & 0xFF == ord('q'):
                  break
          # Release video capture resources
          cap.release()
          cv2.destroyAllWindows()
          `}
      />
      <p>
        We begin by importing necessary libraries: ultralytics (for YOLO object
        detection) and cv2 (OpenCV for video processing). The YOLO class is
        instantiated with the path to the YOLO model file ("../yolov8n.pt")
        which are already downloaded if not just use YOLO("yolov8n.pt") to
        download weights to the curent directory.
        <br />
        <br />
        The video file which you want to track object in it
        (path="object_tracking_yolo/sample.mp4") is loaded using OpenCV’s
        VideoCapture function. The while loop processes each frame from the
        video. cap.read() retrieves the next frame (frame) and a boolean value
        (ret) indicating whether the frame was successfully read.The YOLO model
        is used to detect objects in the frame using model.track(frame,
        persist=True). The persist=True argument ensures that the detected
        objects are tracked across frames.The resulting tracked frame is stored
        in frame_.
        <br />
        <br />
        The tracked frame (frame_) is now displayed using cv2.imshow('video',
        frame_). The window will show the video with bounding boxes around
        detected objects. Pressing the ‘q’ key will exit the loop and close the
        video window. After the loop completes (or when the user presses ‘q’),
        the video capture is released (cap.release()) to free up system
        resources. The OpenCV window is closed using cv2.destroyAllWindows().
        Optionally, From here, we can also take tracked frames and write them
        into an output video file using cv2 VideoWritter.
        <br />
        <br />
      </p>
    </div>
  );
};

export default Content006;
