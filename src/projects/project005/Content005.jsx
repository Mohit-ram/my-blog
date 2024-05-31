import Code from "../../components/Code/Code.jsx";
import mainImg from "./mainImg005.png";

const Content005 = () => {
  return (
    <div className="page container mt-5 mx-auto w-75 px-5 ">
      <h1 className="text-center">Cat Vs Dog</h1>
      <div className="text-center">
        <img src={mainImg} className="h-50 w-50"></img>
      </div>
      <p>
        In this project, we’ll create a machine learning model to classify
        images as either “cat” or “dog.” The process involves data
        preprocessing, where we prepare the dataset for training. Next, we build
        a custom model architecture by adding layers on top of a pre-trained
        InceptionV3 base. This model will be fine-tuned for our specific task.
        After training, we evaluate its performance using validation data and
        visualize the loss. Finally, we assess the model’s accuracy on a test
        dataset. Dataset from Kaggle https://www.kaggle.com/c/dogs-vs-cats.
        
      </p>
      <h4>Data Preprocessing</h4>
      <Code
        code={`
        # Step 1: Data Preprocessing
        # Import necessary libraries
        import os
        import tensorflow as tf
        import numpy as np
        import matplotlib.pyplot as plt
        # Define paths for training and validation datasets
        Train_dir = os.path.join('cats_and_dogs_filtered/train')
        valid_dir = os.path.join('cats_and_dogs_filtered/validation')
        # Create an image dataset from the training directory
        train_ds = tf.keras.utils.image_dataset_from_directory(
            Train_dir,
            labels='inferred',  # Automatically infer labels from subdirectories
            label_mode='binary',  # Treat labels as binary (0 or 1)
            batch_size=64,  # Set batch size for training
            image_size=(150, 150),  # Resize input images to 150x150 pixels
            shuffle=True  # Shuffle the dataset during training
        )
        # Create validation and test datasets from the validation directory
        valid_ds, test_ds = tf.keras.utils.image_dataset_from_directory(
            valid_dir,
            labels='inferred',
            label_mode='binary',
            batch_size=64,
            validation_split=0.5,  # Split validation dataset into halves
            image_size=(150, 150),
            shuffle=True,
            seed=22,
            subset='both'  # Use both validation and test subsets
        )`}
      />
      <p>
        Data Preprocessing: import necessary libraries such as os, tensorflow,
        numpy, and matplotlib. Then define the paths for the training and
        validation datasets (Train_dir and valid_dir) based on the directory
        structure of thecats_and_dogs_filtered” dataset. Next,create an image
        dataset from the training directory using
        tf.keras.utils.image_dataset_from_directory. The parameters used are:
        labels='inferred': Automatically infer labels from subdirectories.
        label_mode='binary': Treat the labels as binary (0 or 1). batch_size=64:
        Set the batch size for training. image_size=(150,150): Resize input
        images to 150x150 pixels. shuffle=True: Shuffle the dataset during
        training. Similarly, create a validation and test datasets from the
        validation directory with similar parameters.
        
      </p>
      <h4>Data Prefetching </h4>
      <Code
        code={`
        # Step 2: Data Preparation
        def preprocessing(data, labels):
            """
            Preprocesses the input data using Inception V3 preprocessing.
            Args:
                data (tf.Tensor): Input image data.
                labels (tf.Tensor): Corresponding labels.
            Returns:
                tf.Tensor: Preprocessed image data.
                tf.Tensor: Unchanged labels.
            """
            data = keras.applications.inception_v3.preprocess_input(data)
            return data, labels
        # Apply preprocessing to the training dataset
        train_ds = train_ds.map(preprocessing, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
        # Apply preprocessing to the validation dataset
        valid_ds = valid_ds.map(preprocessing, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
        # Apply preprocessing to the test dataset
        test_ds = test_ds.map(preprocessing, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
        # Print an example from the training dataset
        for img, label in train_ds.take(1):
            print("Image shape:", img.shape)
            print("Label shape:", label.shape)
        
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
      <h4>Model creation</h4>
      <Code
        code={`
        # Step 3: Model Architecture
        # Load the InceptionV3 pre-trained model (excluding the top classification layers)
        conv_base = keras.applications.InceptionV3(include_top=False, input_shape=(150, 150, 3))
        # Freeze the weights of the pre-trained layers (no training during fine-tuning)
        conv_base.trainable = False
        # Get the output of the 'mixed7' layer from the pre-trained model
        base_last_layer_output = conv_base.get_layer('mixed7').output
        # Flatten the output for further processing
        x = keras.layers.Flatten()(base_last_layer_output)
        # Add a fully connected layer with 512 units
        x = keras.layers.Dense(512)(x)
        # Apply dropout to prevent overfitting
        x = keras.layers.Dropout(0.3)(x)
        # Output layer with sigmoid activation for binary classification
        output = keras.layers.Dense(1, activation='sigmoid')(x)
        # Create the final model by connecting the input to the output
        model = keras.Model(conv_base.input, output)
        # Print a summary of the model architecture
        model.summary()
        `}
      />
      <p> Model summary</p>
      <Code
        code={`
        Model: "model_1"
        __________________________________________________________________________________________________
         Layer (type)                   Output Shape         Param #     Connected to                     
        ==================================================================================================
         input_1 (InputLayer)           [(None, 150, 150, 3  0           []                               
                                        )]                                                                
                                                                                                          
         conv2d (Conv2D)                (None, 74, 74, 32)   864         ['input_1[0][0]']                
                                                                                                          
         batch_normalization (BatchNorm  (None, 74, 74, 32)  96          ['conv2d[0][0]']                 
         alization)                                                                                 
        ...
        Total params: 28,243,873
        Trainable params: 19,268,609
        Non-trainable params: 8,975,264
        `}
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
      <h4> </h4>
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
        We compile the model using the RMSprop optimizer, binary cross-entropy
        loss function, and accuracy metric. The model is then trained using the
        training dataset (train_ds) for 20 epochs, with validation data
        (valid_ds) helping us monitor performance during training. To visualize
        how well the model is learning, we plot the training loss in blue and
        validation loss in blue circles. Lower loss indicates better
        performance. Additionally, higher accuracy means the model predicts
        correctly more often. Finally, we evaluate the model on the test dataset
        (test_ds) and display the test loss and accuracy to assess its
        effectiveness.
        <br />
        <br />
        Model score: [0.4495386481285095, 0.9739999771118164]
      </p>
      <h4>Model Testing</h4>
      <Code
        code={`
        #Step5: Model conclusion and testing on new images
        model.save('my_model.keras')
        my_model = keras.models.load_model('my_model.keras')
        img = tf.io.read_file('VIER PFOTEN_2019-12-13_209-2890x2000-1920x1329.jpg')
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
        output prediction: 1/1 [==============================] - 0s 41ms/step array([[0.]],
        dtype=float32) -- image identified as class 0 which is cat.
        <br />
      </p>
    </div>
  );
};

export default Content005;
