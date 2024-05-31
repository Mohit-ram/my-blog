import Code from "../../components/Code/Code.jsx";
import mainImg from "./mainImg003.png";

const Content003 = () => {
  return (
    <div className="page container mt-5 mx-auto w-75 px-5 ">
      <h1 className="text-center">Digit Recognition</h1>
      <div className="text-center">
        <img src={mainImg} className="h-50 w-50"></img>
      </div>
      <p>
        In this project, we’ll tackle handwritten digit recognition using
        machine learning. The journey includes data preprocessing, building a
        base model, intentional overfitting, regularization, and finding the
        optimal architecture. The data is obtained from the kaggle compettions
        digit-recogniser: https://www.kaggle.com/competitions/digit-recognizer
      </p>
      <h4>Data Preprocessing</h4>
      <Code
        code={`
        # Step 1A: Data Preparation
        import zipfile
        with zipfile.ZipFile('digit-recognizer.zip', 'r') as zip:
            zip.extractall('digit-recogniser')
        import pandas as pd
        data = pd.read_csv('data/train.csv', nrows=5000)
        test = pd.read_csv('data/test.csv', nrows=5000)
        # Step 1B: Data Visualization
        import seaborn as sns
        # Check for missing values
        data.isnull().any().describe()
        # Create a count plot for the 'label' column
        sns.countplot(x=data['label'])
        # Step 1C: Numeric Data
        # Separate features (pixel values) and labels
        x_train = data.drop('label', axis=1)
        y_train = data['label']
        # Normalize pixel values to [0, 1]
        x_train = x_train / 255.
        test = test / 255.
        # Reshape features into 28x28 images
        x_train = x_train.values.reshape(-1, 28, 28, 1)
        test = test.values.reshape(-1, 28, 28, 1)
        # One-hot encode labels
        from tensorflow.keras.utils import to_categorical
        y_train = to_categorical(y_train)
        # Split data into training and validation sets
        from sklearn.model_selection import train_test_split
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2)
          
          `}
      />
      <p>
        In this initial step, we start by extracting the contents of the
        ‘digit-recognizer.zip’ file. This archive likely contains the training
        and test data for our handwritten digit recognition task. We use the
        zipfile module to extract the files, and then load the first 5000 rows
        of the training and test data using Pandas. Next, we perform some
        exploratory data analysis. We check if there are any missing values in
        the dataset. The line data.isnull().any().describe() provides a summary
        of whether each column contains missing values. Additionally, we create
        a count plot using Seaborn to visualize the distribution of the ‘label’
        column. This column represents the digit labels (ranging from 0 to 9)
        c/corresponding to the handwritten images.
        <br />
        <br />
        Now, let’s separate the features (pixel values) and labels (digit
        classes) from the training data. We assign the pixel values to x_train
        and the corresponding labels to y_train.To ensure consistent training,
        we normalize the pixel values to a range between 0 and 1 by dividing
        them by 255. This step helps improve convergence during model training.
        We reshape the data into 28x28 images, as each handwritten digit image
        is represented by a 28x28 grid of pixels.Finally, we one-hot encode the
        labels for training. This means converting the single-digit labels (0 to
        9) into binary vectors. For example, the label ‘3’ becomes [0, 0, 0, 1,
        0, 0, 0, 0, 0, 0].Additionally, we split the data into training and
        validation sets using train_test_split. This ensures that we have a
        separate validation subset to evaluate our model during training
      </p>
      <div className="container text-center">
        <p>Seaborn Count plot of all digits in dataset</p>
        <img
          className="h-50 w-50"
          src="/my-blog/src/projects/project003/img01.png"
        />
      </div>
      <h4>Base Model</h4>
      <Code
        code={`
        #Step2: Base Model Architecture
        from tensorflow import keras
        from tensorflow.keras import layers
        # Define input layer with shape (28, 28, 1) for grayscale images
        inputs = keras.Input((28, 28, 1))
        # Add two convolutional layers with 64 filters, kernel size 5x5, and ReLU activation
        x = layers.Conv2D(filters=64, kernel_size=5, activation='relu', padding='same')(inputs)
        x = layers.Conv2D(filters=64, kernel_size=5, activation='relu', padding='same')(x)
        # Add a max-pooling layer with pool size 2x2
        x = layers.MaxPool2D(pool_size=2)(x)
        # Flatten the output from the convolutional layers
        x = layers.Flatten()(x)
        # Add a fully connected (dense) layer with 256 units and ReLU activation
        x = layers.Dense(256, activation='relu')(x)
        # Output layer with 10 units (one for each digit class) and softmax activation
        outputs = layers.Dense(10, activation='softmax')(x)
        # Create the model
        model = keras.Model(inputs=inputs, outputs=outputs)
        # Print a summary of the model architecture
        model.summary()`}
      />
      <p>
        Step 2: Base Model Architecture In this step, we define the architecture
        of our neural network model. We’ll build a simple convolutional neural
        network (CNN) for digit classification. Here’s a breakdown of the code.
        We start with an input layer that expects 28x28 grayscale images (1
        channel). Two convolutional layers with 64 filters each are added. These
        layers learn features from the input images. A max-pooling layer reduces
        the spatial dimensions of the feature maps. The flattened output is
        passed to a dense layer with 256 units (neurons) and ReLU activation.
        Finally, the output layer has 10 units (for digits 0 to 9) with softmax
        activation for classification. This base model serves as a starting
        point, and you can further customize it or add more layers as needed.
      </p>
      <h4>Model Evaluation</h4>
      <Code
        code={`
        # Step 3: Model Configuration and Evaluation
        # Configure the model
        model.compile(optimizer='rmsprop',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        # - We set the optimizer to 'rmsprop', which adapts the learning rate during training.
        # - The loss function is 'categorical_crossentropy' for multi-class classification.
        # - We track accuracy as our evaluation metric.
        # Train the model
        history = model.fit(x_train, y_train, epochs=10, batch_size=128, validation_data=(x_val, y_val))
        # Model evaluation
        import matplotlib.pyplot as plt
        loss_values = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(1, len(loss_values) + 1)
        plt.plot(epochs, loss_values, 'bo', label="Training Loss")
        plt.plot(epochs, val_loss, 'b', label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.show()
          
          `}
      />
      <p>
        we configure the neural network model by specifying the optimizer, loss
        function, and evaluation metric. The optimizer is set to ‘rmsprop’,
        which adapts the learning rate during training. The loss function is
        ‘categorical_crossentropy’, suitable for multi-class classification. We
        track accuracy as the evaluation metric. Next, we train the model using
        the training data (x_train and y_train). We train for 10 epochs
        (iterations over the dataset) with a batch size of 128. Validation data
        (x_val and y_val) are used to evaluate the model during training.
        Finally, we plot the training loss and validation loss over the epochs,
        helping us monitor how well the model is learning and detect
        overfitting.
      </p>
      <h4>Overfitting</h4>
      <Code
        code={`
        #Step4: Overfiting the model
        # Added more conv layers
        inputs2 = keras.Input((28,28,1))
        x = layers.Conv2D(filters=32, kernel_size=5, activation='relu', padding='same')(inputs2)
        x = layers.Conv2D(filters=32, kernel_size=5, activation='relu', padding='same')(x)
        x = layers.MaxPool2D(pool_size=2)(x)
        x = layers.Conv2D(filters=64, kernel_size=5, activation='relu', padding='same')(x)
        x = layers.Conv2D(filters=64, kernel_size=5, activation='relu', padding='same')(x)
        x = layers.MaxPool2D(pool_size=2)(x)
        x = layers.Conv2D(filters=128, kernel_size=5, activation='relu', padding='same')(x)
        x = layers.Conv2D(filters=128, kernel_size=5, activation='relu', padding='same')(x)
        x = layers.MaxPool2D(pool_size=2)(x)
        x = layers.Flatten()(x)
        x= layers.Dense(256, activation='relu')(x)
        outputs2 = layers.Dense(10, activation='softmax')(x)
        model2 = keras.Model(inputs = inputs2, outputs= outputs2)
        model2.summary()
          
          `}
      />
      <p>
        our mission is to intentionally overfit the model. We’ll achieve this by
        adding more layers, increasing the number of filters, extending the
        training epochs, and introducing additional dense layers. Our goal is to
        observe how far we can push the model before the validation accuracy
        becomes unstable. As we make these modifications, we’ll closely monitor
        the validation accuracy, ensuring that we strike the right balance
        between performance on the training data and generalization to unseen
        data
      </p>
      <h4>Regularisation</h4>
      <Code
        code={`
        #step5:regularising and the final model
        #using drop out layers to regularise model
        #dropping conv layer to prevent overfitting
        inputs3 = keras.Input((28,28,1))
        x = layers.Conv2D(filters=32, kernel_size=5, activation='relu', padding='same')(inputs3)
        x = layers.Conv2D(filters=32, kernel_size=5, activation='relu', padding='same')(x)
        x = layers.MaxPool2D(pool_size=2)(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Conv2D(filters=64, kernel_size=5, activation='relu', padding='same')(x)
        x = layers.Conv2D(filters=64, kernel_size=5, activation='relu', padding='same')(x)
        x = layers.MaxPool2D(pool_size=2, strides=2)(x)
        x = layers.Dropout(0.25)(x)
        x = layers.Flatten()(x)
        x= layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        outputs3 = layers.Dense(10, activation='softmax')(x)
        model3 = keras.Model(inputs = inputs3, outputs= outputs3)
        model3.summary()
          
          `}
      />
      <p>
        we focus on regularization techniques to enhance our model’s performance
        and mitigate overfitting. By introducing dropout layers, selectively
        adjusting the number of convolutional layers, and fine-tuning the
        architecture, we aim to strike the right balance between training
        accuracy and generalization.
      </p>
      <h4>Model Performance </h4>
      <div className="container d-flex-inline text-center">
        <img
          className="h-25 w-25"
          src="/my-blog/src/projects/project003/img02.png"
        />
        <img
          className="h-25 w-25"
          src="/my-blog/src/projects/project003/img03.png"
        />
        <img
          className="h-25 w-25"
          src="/my-blog/src/projects/project003/img04.png"
        />
      </div>

      <pre>
        #step6: Model conclusion model3.save('model.keras') neural_model=
        keras.models.load_model('model.keras')
      </pre>
    </div>
  );
};

export default Content003;
