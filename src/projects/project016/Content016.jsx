import Code from "../../components/Code/Code.jsx";
import img01 from"./output.png";

const Content016 = () => {
  return (
    <div className="page container mt-5 mx-auto w-75 px-5 ">
      <h1 className="text-center">NLP: Sarcasm or Not</h1>
      <div className="text-center"></div>
      <p>
        This project focuses on developing a machine learning model to detect
        sarcasm in text using a neural network. The model is trained on a
        dataset of headlines labeled as sarcastic or not. It involves data
        preprocessing, model training, and evaluation. Additionally, a
        user-friendly web application is created using Streamlit, allowing users
        to input sentences and receive real-time predictions on whether the text
        is sarcastic. This application leverages the power of deep learning to
        understand and classify nuanced human language, providing a practical
        tool for sentiment analysis.
      </p>
      <h4></h4>
      <Code
        code={`
          # Step 1: Data Loading
            import tensorflow as tf
            from tensorflow import keras
            import numpy as np
            import pandas as pd
            import matplotlib.pyplot as plt
            from tensorflow.keras.preprocessing.sequence import pad_sequences
            import json
            import urllib

            #run to download sarcasm.json once
            url = 'https://storage.googleapis.com/download.tensorflow.org/data/sarcasm.json'
            urllib.request.urlretrieve(url, 'sarcasm.json')
            # Load data from JSON file
            with open('sarcasm.json', 'r') as file:
                data = json.load(file)
            # Initialize lists for sentences and labels
            sentences = []
            labels = []
            # Extract sentences and labels from the data
            for row in data:
                sentences.append(row['headline'])
                labels.append(row['is_sarcastic'])
            # Step 2: Data Preprocessing
            # Define parameters for tokenization and padding
            vocab_size = 1000  # Vocabulary size
            oov_token = '<OOV>'  # Token for out-of-vocabulary words
            max_word_length = 60  # Maximum length of sequences
            padding = 'post'  # Padding type (add padding at the end)
            trunc_type = 'post'  # Truncation type (truncate at the end)
            # Initialize the tokenizer
            tokeniser = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_size, oov_token=oov_token)
            # Fit the tokenizer on the sentences
            tokeniser.fit_on_texts(sentences)
            # Convert sentences to sequences of integers
            sequences = tokeniser.texts_to_sequences(sentences)
            # Pad sequences to ensure uniform length
            padded_seq = pad_sequences(sequences, maxlen=max_word_length, padding=padding, truncating=trunc_type)
            #Step3: Data Splits 
            split_size = int(0.7*len(sentences))
            train_seq = padded_seq[:split_size]
            val_seq = padded_seq[split_size: split_size+int(0.15*len(sentences))]
            test_seq = padded_seq[split_size+int(0.15*len(sentences)):]
            print([val.shape for val in [train_seq, val_seq, test_seq]])
            train_lab = np.array(labels[:split_size], dtype=np.float32)
            val_lab = np.array(labels[split_size: split_size+int(0.15*len(sentences))], dtype=np.float32)
            test_lab = np.array(labels[split_size+int(0.15*len(sentences)):], dtype=np.float32)
            print([len(val) for val in [train_lab, val_lab, test_lab]])
          
          `}
      />
      <p>
        <h5>Step 1: Data Loading</h5>
        The code begins by importing necessary libraries such as TensorFlow,
        Keras, NumPy, Pandas, and Matplotlib. These libraries are essential for
        building and training machine learning models, handling data, and
        visualizing results. The json library is used to load data from a JSON
        file named sacrasm.json. The data consists of headlines and their
        corresponding labels indicating whether they are sarcastic or not. The
        headlines are stored in the sentences list, and the labels are stored in
        the labels list.
        <h5>Step 2: Data Preprocessing</h5>
        In this step, the text data is converted into a numeric format suitable
        for machine learning models. The vocabulary size is set to 1000, meaning
        only the top 1000 most frequent words will be considered. An
        out-of-vocabulary token (OOV) is used to handle words not in the
        vocabulary. The maximum length of each sequence is set to 60 words, and
        sequences longer than this will be truncated. The Tokenizer class from
        Keras is used to tokenize the text data, converting each word into a
        corresponding integer. These sequences are then padded to ensure they
        all have the same length, which is necessary for batch processing in
        neural networks. In step3, the dataset is divided into three subsets:
        training, validation, and testing. This is a common practice in machine
        learning to ensure that the model is trained, validated, and tested on
        different portions of the data, which helps in evaluating the model’s
        performance and generalization ability. First, the code calculates the
        split size for the training set, which is 70% of the total data. The
        training sequences (train_seq) are then extracted from the padded
        sequences up to this split size. The validation sequences (val_seq) are
        taken from the next 15% of the data, and the testing sequences
        (test_seq) are taken from the remaining 15%.
        <br />
        <br />
      </p>
      <h4>Model creation</h4>
      <Code
        code={`
          # Step 4: Model Architecture
          # Define the input layer with a fixed sequence length
          inputs = keras.layers.Input(shape=(max_word_length,))
          # Add an embedding layer to convert words to dense vectors
          x = keras.layers.Embedding(input_dim=vocab_size, output_dim=16, input_length=max_word_length)(inputs)
          # Add a bidirectional LSTM layer to capture dependencies in both directions
          x = keras.layers.Bidirectional(keras.layers.LSTM(64))(x)
          # Add a dense layer with ReLU activation
          x = keras.layers.Dense(24, activation='relu')(x)
          # Add a dropout layer to prevent overfitting
          x = keras.layers.Dropout(0.4)(x)
          # Define the output layer with sigmoid activation for binary classification
          outputs = keras.layers.Dense(1, activation='sigmoid')(x)
          # Create the model
          model = keras.Model(inputs, outputs)
          # Print the model summary
          model.summary()

          # Step 5: Model Training
          # Define a callback to save the best model based on validation performance
          callbacks_list = [keras.callbacks.ModelCheckpoint('sarcasm_model.h5', save_best_only=True)]
          # Compile the model with Adam optimizer and binary cross-entropy loss
          model.compile(optimizer=keras.optimizers.Adam(),
                        loss='binary_crossentropy',
                        metrics=['accuracy'])
          # Train the model on the training data with validation
          history = model.fit(train_seq, train_lab,
                              epochs=10,
                              validation_data=(val_seq, val_lab),
                              callbacks=callbacks_list)

          # Step 6: Model Conclusion
          # Plot the training and validation loss
          train_loss = history.history['loss']
          val_loss = history.history['val_loss']
          plt.plot(train_loss, 'b')
          plt.plot(val_loss, 'bo')
          # Load the best saved model
          loaded_model = keras.models.load_model('sarcasm_model.h5')
          # Evaluate the current model on the validation set
          scores = model.evaluate(val_seq, val_lab)  #[0.39555516839027405, 0.8217673301696777]
          # Evaluate the loaded model on the validation set
          new_scores = loaded_model.evaluate(val_seq, val_lab) #[0.3617543876171112, 0.8382426500320435]
         `}
      />
      <p>
        <h5>Step 4: Model Architecture</h5> The model architecture is defined using
        Keras. It starts with an input layer that accepts sequences of a fixed
        length. An embedding layer is used to convert these sequences into dense
        vectors of fixed size. A bidirectional LSTM layer follows, which helps
        in capturing dependencies in both forward and backward directions. A
        dense layer with ReLU activation is added, followed by a dropout layer
        to prevent overfitting. Finally, the output layer uses a sigmoid
        activation function to produce a probability score for binary
        classification.
        <br />
        <h5>Step 5: Model Training</h5> The model is compiled with the Adam optimizer and
        binary cross-entropy loss function, suitable for binary classification
        tasks. The training process involves fitting the model on the training
        data for 10 epochs, with validation on a separate validation set. A
        callback is used to save the best model based on validation performance.
        <br />
        <h5>Step 6: Model Conclusion</h5> After training, the loss values for both
        training and validation sets are plotted to visualize the training
        process. The best model is loaded, and its performance is evaluated on
        the validation set. The evaluation scores are printed to compare the
        performance of the saved model with the current model.
        <br />
        <br />
      </p>
      <div className="text-center">
        <p>Train vs Val Loss curve</p>
        <img
          src={img01}
          alt="result1"
          style={{ height: "300px", width: "300px" }}
        />
</div>
      <h4>Deploy app.py using streamlit</h4>
      <Code
        code={`
          #streamlit App
          import tensorflow as tf
          from tensorflow.keras.models import load_model
          from tensorflow.keras.utils import pad_sequences
          import json
          with open('sarcasm.json', 'r') as file:
              data = json.load(file)
              sentences = []
          for row in data:
              sentences.append(row['headline'])
              
          def preprocess_data(text):
              vocab_size = 1000
              oov_token = '<OOV>'
              max_word_length = 60
              padding='post'
              trunc_type = 'post'
              #BATCH_SIZE = int(len(sentences)/2)              
              tokeniser = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_size, oov_token=oov_token)
              tokeniser.fit_on_texts(sentences)
              #convert into sequences
              sequence = tokeniser.texts_to_sequences(text)              
              #pad sequences
              padded_seq = pad_sequences(sequence,
                                          maxlen=max_word_length,
                                          padding=padding,
                                      truncating = trunc_type)
              
              encoded_text = padded_seq
              return encoded_text
          model = load_model('sarcasm_model.h5')

          import streamlit as st
          st.title("Sarcastic or not")
          st.write("Enter a sentence to classify as sarcastic or not.")
          user_input = st.text_area("Your sentence")
          if st.button('yes or no'):
              encoded_text = preprocess_data(user_input)
              prediction = model.predict(encoded_text)
              answer = 'Yes' if prediction[0][0] >0.5 else 'No'
              st.write(f'Sarcastic: {answer}')
              st.write(f'Prediction Score: {prediction[0][0]}')
          else:
              st.write('Please enter a movie review.')           
          `}
      />
      <p>
        The main.py above code creates a web application using Streamlit to
        classify sentences as sarcastic or not. The application loads a
        pre-trained sarcasm detection model and allows users to input sentences
        for classification.
        <br />
        Preprocessing Function: A function preprocess_data is defined to
        preprocess the input text. This function tokenizes the text, converts it
        into sequences, and pads the sequences to ensure uniform length. The
        tokenizer is fitted on the sentences from the dataset. The pre-trained
        sarcasm detection model is loaded using load_model.The Streamlit
        application is defined and It includes a title and a text area for user
        input. When the user clicks the button, the input text is preprocessed,
        and the model predicts whether the sentence is sarcastic or not. The
        result and prediction score are displayed.
        <br />
      </p>
    </div>
  );
};

export default Content016;
