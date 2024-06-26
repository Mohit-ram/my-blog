import Code from "../../components/Code/Code.jsx";
import mainImg from "./mainImg004.png";
import out01 from "./t_vs_v.png";

const Content004 = () => {
  return (
    <div className="page container mt-5 mx-auto w-75 px-5 ">
      <h1 className="text-center">Horse or Human</h1>
      <div className="text-center">
        <img src={mainImg} className="h-50 w-50"></img>
      </div>
      <p>
        In this project, we build and evaluate a binary image classification
        model using transfer learning. We leverage the pre-trained InceptionV3
        architecture, preprocess the data, train the model, and assess its
        performance. Finally, we demonstrate how to use the model for
        predictions on unseen images. The dataset is gracefully provided by
        https://laurencemoroney.com/datasets.html
      </p>
      <h4>Data Preprocessing</h4>
      <Code
        code={`            
        # Step 1: Data Preprocessing
        import os
        import tensorflow as tf
        from tensorflow import keras
        import numpy as np
        # Define paths for training and validation data directories
        Train_dir = os.path.join('train')
        valid_dir = os.path.join('validation')
        # Create TensorFlow datasets using image_dataset_from_directory
        train_ds = tf.keras.utils.image_dataset_from_directory(
            Train_dir,
            labels='inferred',  # Automatically infer labels from subdirectories
            label_mode='binary',  # Binary classification (you can change this if needed)
            batch_size=64,
            image_size=(150, 150),  # Resize images to 150x150 pixels
            shuffle=True  # Shuffle the dataset
        )
        valid_ds = tf.keras.utils.image_dataset_from_directory(
            valid_dir,
            labels='inferred',
            label_mode='binary',
            batch_size=64,
            image_size=(150, 150),
            shuffle=True
        )
        # Ensure that the 'train' and 'validation' directories contain labeled image data
      `}
      />
      <p>
        Import the necessary libraries: os, tensorflow, and numpy. We define the
        paths for the training and validation data directories (Train_dir and
        valid_dir) which are exsiting in same directory when downloaded. Next,
        we create TensorFlow datasets using image_dataset_from_directory. The
        image_dataset_from_directory function in TensorFlow is a convenient
        utility for creating a tf.data.Dataset from image files stored in a
        directory.EW provide the path to a directory containing subdirectories,
        where each subdirectory corresponds to a different class or label. For
        example, if you have two classes (“class_a” and “class_b”)
        <br />
        The function reads images from these subdirectories and automatically
        assigns labels based on the subdirectory names (e.g., “class_a” gets
        label 0, “class_b” gets label 1). It returns a tf.data.Dataset that
        yields batches (here 64) of images along with their corresponding
        labels.
      </p>
      <h4>Data prefetching</h4>
      <Code
        code={`
          
        # Step 2: Data Preparation
        # Define a preprocessing function
        def preprocessing(data, labels):
            # Preprocess input data using Inception V3 model's preprocessing
            data = keras.applications.inception_v3.preprocess_input(data)
            return data, labels
        # Apply preprocessing to training and validation dataset
        train_ds = train_ds.map(preprocessing, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
        valid_ds = valid_ds.map(preprocessing, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
        # Print a batch of data from the training dataset
        for img, label in train_ds.take(1):
            print("Preprocessed Images Shape:", img.shape)
            print("Labels Shape:", label.shape)
            `}
      />

      <p>
        Data Preprocessing Function (preprocessing): The preprocessing function
        is defined to process the input data and labels.It takes two arguments:
        data (representing images) and labels. The function preprocesses the
        input data using the Inception V3 model’s preprocessing function into a
        type that Inception model needs.The preprocessed data is returned along
        with the original labels.This step ensures that the input data is
        properly formatted and ready for model training.
        <br />
        <br />
        Mapping and Parallelization: The map function is applied to both the
        training (train_ds) and validation (valid_ds) datasets.It applies the
        preprocessing function to each batch of data in
        parallel.num_parallel_calls=tf.data.AUTOTUNE allows TensorFlow to
        optimize the parallelization based on available resources. The
        preprocessed datasets are then stored in train_ds and
        valid_ds.Prefetching. The prefetch operation is applied to both
        datasets.It prefetches batches of data to improve training performance
        by overlapping data loading and model execution. tf.data.AUTOTUNE
        dynamically adjusts the prefetch buffer size based on available memory
        and CPU resources.
        <br />
        <br />
        Inspecting Data: The code snippet at the end prints a batch of data from
        train_ds.for img, label in train_ds.take(1): iterates over the first
        batch.img contains the preprocessed images, and label contains their
        corresponding labels.The shape of the labels is printed using
        label.shape.
      </p>
      <h4>Model architecture</h4>
      <Code
        code={`
        # Step 3: Model Architecture
        # Create an InceptionV3 base model without top layers
        conv_base = keras.applications.InceptionV3(include_top=False, input_shape=(150, 150, 3))
        conv_base.trainable = False  # Freeze the base model weights
        # Extract output from a specific layer ('mixed7')
        base_last_layer_output = conv_base.get_layer('mixed7').output
        # Flatten the output for fully connected layers
        x = keras.layers.Flatten()(base_last_layer_output)
        # Add a dense layer with 512 units
        x = keras.layers.Dense(512)(x)
        # Apply dropout regularization
        x = keras.layers.Dropout(0.3)(x)
        # Final output layer for binary classification
        output = keras.layers.Dense(1, activation='sigmoid')(x)
        # Create the complete model
        model = keras.Model(conv_base.input, output)
        # Print model summary
        model.summary()`}
      />
      <p>
        The code sets up a transfer learning approach by using a pre-trained
        InceptionV3 base model.The flattened features are passed through
        additional layers to create a custom classifier.The resulting model is
        suitable for binary classification tasks. InceptionV3 Model: The
        InceptionV3 architecture is a deep convolutional neural network (CNN)
        designed for image classification tasks. It builds upon the original
        Inception model (also known as GoogLeNet) and introduces several
        improvements. InceptionV3 is part of the Inception family and has been
        widely used due to its efficiency and accuracy1.The architecture uses a
        series of convolutional layers to extract features from input
        images.These layers apply filters to the input data, capturing different
        patterns and structures.Pooling layers reduce the dimensions of feature
        maps while preserving important information.In InceptionV3, max pooling
        and average pooling are commonly used.The core innovation of Inception
        models lies in their inception modules.
        <br />
        <br />
        An inception module combines multiple parallel convolutional layers with
        different filter sizes. By doing so, it captures features at various
        scales and resolutions. The inception module helps prevent overfitting
        and improves model performance. InceptionV3 includes auxiliary
        classifiers to propagate label information deeper into the network.
        These classifiers aid in training and regularization.Batch normalization
        is applied to layers in the sidehead of the network. It helps stabilize
        training by normalizing the input to each layer2.
      </p>

      <h4>Model Evaluation</h4>
      <Code
        code={`
        # Step 4: Model Evaluation
        # Compile the model with optimizer, loss function, and evaluation metric
        model.compile(optimizer='rmsprop',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        # Train the model using the training dataset
        history = model.fit(train_ds, epochs=20, validation_data=valid_ds)
        # Visualize training and validation loss
        train_loss = history.history['loss']
        val_loss = history.history['val_loss']
        plt.plot(train_loss, 'b', label='Training Loss')
        plt.plot(val_loss, 'bo', label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        # Evaluate the model on the test dataset
        scores = model.evaluate(test_ds)
        print(f"Test Loss: {scores[0]:.4f}, Test Accuracy: {scores[1]*100:.2f}%")
         `}
      />
      <p>
        we compiled our model by specifying the optimizer (RMSProp), loss
        function (binary cross-entropy), and accuracy metric. Then, we trained
        the model using the provided datasets for 20 epochs. During training,
        the model’s weights were updated based on the loss and accuracy.
        Finally, we visualized the training and validation loss curves to assess
        model performance.
      </p>
      <div className="text-center">
      <p> Training vs validation loss</p>
      <img
        className="text-center h-50 w-50"
        src={out01}
        alt="Train vs val accuracy"
      ></img>
      </div>

      <h4>Model Conclusion</h4>
      <Code
        code={`
        #Step5: Model Conclusion and test
        model.save('my_model.keras')
        model_loaded = keras.models.load_model('my_model.keras')
        #test model with never seen image
        img = tf.io.read_file('image_name.jpg')
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, size=(150,150))
        img, lab = preprocessing(img,1)
        img =  tf.expand_dims(img, axis=0)
        img.shape
        pred = model.predict(img)
        pred
        `}
      />
      <p>
        we save the trained model using model.save('my_model.keras'), creating a
        file that encapsulates the model’s architecture, weights, and optimizer
        configuration. Later, we load this saved model using
        keras.models.load_model('my_model.keras'), effectively restoring the
        entire model for further use. To test the model’s performance, we
        process an unseen image: reading the image file, decoding it, resizing
        it to the desired input size (150x150 pixels), applying preprocessing
        steps (such as normalization), and expanding its dimensions to create a
        batch input. Finally, we make predictions using model.predict(img),
        obtaining the model’s confidence in binary classification (e.g., whether
        the image belongs to a specific class). Remember to ensure the image
        file exists and consider using a descriptive filename for better
        organization.
        <br />
        <br />
        output: 1/1 [==============================] - 0s 41ms/step
        array([[0.]], dtype=float32) -- human
        <br />
      </p>
    </div>
  );
};

export default Content004;
