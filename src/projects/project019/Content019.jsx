import Code from "../../components/Code/Code.jsx";
import img01 from "./img01.png";
import img02 from "./img02.png";

const Content019 = () => {
  return (
    <div className="page container mt-5 mx-auto w-75 px-5 ">
      <h1 className="text-center">NLP: Text Classification</h1>
      <div className="text-center"></div>
      <p>
        This project focuses on developing a text classification model using
        TensorFlow and Keras to categorize BBC News articles into predefined
        categories(Business, Sports, Politics, ...). The process involves several key steps, starting with data
        preprocessing, followed by model definition and training, and concluding
        with model evaluation. The dataset used is obtained form kaggle
        competetions at https://www.kaggle.com/c/learn-ai-bbc .
      </p>

      <h5>Data Preprocessing</h5>
      <Code
        code={`
          #Step1: DataPreprocesing
          import os
          import tensorflow as tf
          from tensorflow import keras
          import pandas as pd
          import numpy as np
          import matplotlib.pyplot as plt
          import csv

          # Data loading
          train_df = pd.read_csv('BBC News Train.csv', usecols=['Text', 'Category'])
          test_df = pd.read_csv('BBC News Test.csv', usecols=['Text'])

          # Data transformation
          from tensorflow.keras.preprocessing.text import Tokenizer
          from tensorflow.keras.preprocessing.sequence import pad_sequences

          # Initialize tokenizer with a vocabulary size of 1000 and an out-of-vocabulary token
          tokenizer = Tokenizer(num_words=1000, oov_token='<OOV>')
          def text_to_numeric(text, max_len, num_words=1000):
              # Fit tokenizer on the text data
              tokenizer = Tokenizer(num_words=num_words, oov_token='<OOV>')
              tokenizer.fit_on_texts(text)
              # Convert text to sequences of integers
              sequences = tokenizer.texts_to_sequences(text)
              # Pad sequences to ensure uniform length
              padded_sequence = pad_sequences(sequences, maxlen=max_len, padding='post')
              return np.array(padded_sequence)

          # Convert training and testing text data to numerical format
          train_sequences = text_to_numeric(train_df['Text'], max_len=280)
          train_labels = text_to_numeric(train_df['Category'], max_len=1) - 1
          x_test = text_to_numeric(test_df['Text'], max_len=280)

          # Split the data into training and validation sets (80% training, 20% validation)
          split_size = int(0.8 * train_sequences.shape[0])
          x_train = train_sequences[:split_size]
          y_train = train_labels[:split_size]
          x_val = train_sequences[split_size:]
          y_val = train_labels[split_size:]

          # Print the shapes of the resulting datasets
          print([p.shape for p in [x_train, x_val, y_train, y_val]])
          #output: [(1192, 280), (298, 280), (1192, 1), (298, 1)]
          # Print the tokenizer's word index
          print(tokenizer.word_index)
          # output: {'<OOV>': 1, 'sport': 2, 'business': 3, 'politics': 4, 'entertainment': 5, 'tech': 6}

          
          `}
      />
      <p>
        we begin with importing necessary libraries such as TensorFlow, Keras,
        Pandas, NumPy, and Matplotlib. The data is then loaded from CSV files
        into Pandas DataFrames, specifically selecting the ‘Text’ and ‘Category’
        columns for training data and ‘Text’ for testing data.
        <br />
        The next step involves transforming the text data into numerical format
        using the Keras Tokenizer. The text_to_numeric function is defined to
        convert text into sequences of integers, which are then padded to ensure
        uniform length. This function is applied to both the training and
        testing datasets. The training labels are also converted to numerical
        format and adjusted to start from zero.
        <br />
        The data is then split into training and validation sets. The split is
        done by taking 80% of the data for training and the remaining 20% for
        validation. Finally, the shapes of the resulting datasets are printed to
        verify the preprocessing steps. The tokenizer’s word index is also
        printed to understand the mapping of words to integers
      </p>
      <h5>Define Model</h5>
      <Code
        code={`
          # Model architecture
          inputs = keras.Input(shape=(280,))  # Define input layer with sequence length 280
          x = keras.layers.Embedding(input_dim=1000, output_dim=16, input_length=280)(inputs)  # Embedding layer
          x = keras.layers.GlobalAveragePooling1D()(x)  # Global Average Pooling layer
          x = keras.layers.Dense(24, activation='relu')(x)  # Dense layer with ReLU activation
          outputs = keras.layers.Dense(6, activation='softmax')(x)  # Output layer with softmax activation
          model = keras.Model(inputs, outputs)  # Define the model
          model.summary()  # Print model summary

          # Model Training
          model.compile(optimizer=keras.optimizers.Adam(),  # Compile the model with Adam optimizer
                        loss='sparse_categorical_crossentropy',  # Use sparse categorical cross-entropy loss
                        metrics=['accuracy'])  # Evaluate model using accuracy metric

          history = model.fit(x_train, y_train,  # Train the model
                              epochs=50, batch_size=16,  # Set number of epochs and batch size
                              validation_data=(x_val, y_val))  # Use validation data for evaluation

          # Plot training and validation loss
          train_loss = history.history['loss']
          val_loss = history.history['val_loss']
          plt.plot(train_loss, 'b', label='Training Loss')
          plt.plot(val_loss, 'bo', label='Validation Loss')
          plt.legend()
          plt.show()          
          `}
      />
      <p>
        The model architecture begins with an input layer that accepts sequences
        of length 280. This input is then passed through an embedding layer,
        which converts the input sequences into dense vectors of fixed size (16
        in this case). The embedding layer helps in capturing the semantic
        meaning of the words.
        <br />
        Following the embedding layer, a Global Average Pooling layer is
        applied. This layer reduces the dimensionality of the input by averaging
        the values across the sequence, effectively summarizing the information.
        Next, a Dense layer with 24 units and ReLU activation function is added.
        This layer introduces non-linearity to the model, allowing it to learn
        more complex patterns. The final layer is a Dense layer with 6 units and
        a softmax activation function, which outputs a probability distribution
        over the 6 categories.
        <br />
        The model is then compiled with the Adam optimizer, sparse categorical
        cross-entropy loss function, and accuracy as the evaluation metric. The
        model is trained on the training data for 50 epochs with a batch size of
        16. During training, the model’s performance is also evaluated on the
        validation set. The training and validation loss values are stored and
        plotted to visualize the training process.
      </p>
      <div className="text-center">
        <p>Train and Val loss VS epochs</p>
        <img
          src={img01}
          alt="result1"
          style={{ height: "300px", width: "300px" }}
        />
      </div>
      <h5>Model Conclusion</h5>
      <Code
        code={`
          #Step3:Model Evaluation
          scores = model.evaluate(x_val, y_val)  # Evaluate the model on validation data
          print(scores)  #[0.16155174374580383, 0.9563758373260498]

          # Model Predictions
          preds = model.predict(x_val)  # Predict the classes for validation data
          predictions = np.array([np.argmax(pred) for pred in preds])  # Convert predictions to class labels

          # Create a DataFrame to compare actual and predicted labels
          preds_df = pd.DataFrame({'actual': y_val.flatten(), 'predictions': predictions})

          # Calculate accuracy manually
          accuracy = len(preds_df[preds_df['actual'] == preds_df['predictions']]) / len(y_val)
          print(accuracy)  #0.9563758389261745
          
          `}
      />

      <div className="text-center">
        <p>actul vs predictions DF</p>
        <img
          src={img02}
          alt="result1"
          style={{ height: "300px", width: "200px" }}
        />
      </div>
      <p>
        The model.evaluate function is used to compute the loss and accuracy of
        the model on the validation data, returning a list of scores. The first
        element of this list is the loss, and the second element is the
        accuracy. Next, the model’s predictions on the validation data are
        obtained using model.predict. These predictions are then converted into
        class labels by taking the index of the maximum value in each prediction
        array. A DataFrame is created to compare the actual labels with the
        predicted labels. Finally, the accuracy of the model is calculated
        manually by comparing the actual and predicted labels. The proportion of
        correct predictions is computed by dividing the number of correct
        predictions by the total number of validation samples.
        <br />
        Acurracy of the model = 0.956 = 96%
        <br />
      </p>
    </div>
  );
};

export default Content019;
