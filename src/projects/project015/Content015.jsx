import Code from "../../components/Code/Code.jsx";


const Content015 = () => {
  return (
    <div className="page container mt-5 mx-auto w-75 px-5 ">
      <h1 className="text-center">NLP: Sentiment Analysis</h1>
      <div className="text-center">
        
      </div>
      <p>
        This project focuses on building and deploying a sentiment analysis
        model using the IMDB movie reviews dataset. Sentiment analysis is a
        natural language processing (NLP) technique used to determine whether a
        piece of text is positive, negative, or neutral. In this project, we aim
        to classify movie reviews as either positive or negative.
      </p>
      <h4>Step1 Model Creation </h4>
      <h5>Step1a: Data Preprcessing</h5>

      <Code
        code={`
          # Import necessary libraries
            import numpy as np 
            import tensorflow as tf
            from tensorflow.keras.datasets import imdb

            # Set the vocabulary size to 5000
            vocab_size = 5000

            # Load the IMDB dataset with the specified vocabulary size
            (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size)

            # Import the pad_sequences function
            from tensorflow.keras.utils import pad_sequences
            # Set the maximum length for the input 
            max_length = 250
            # Pad the training sequences to ensure uniform length
            x_train_pad = pad_sequences(x_train, max_length)
            # Pad the test sequences to ensure uniform length
            x_test_pad = pad_sequences(x_test, max_length)
          `}
      />
      <p>
        The popular IMDB dataset, which is commonly used for sentiment analysis
        tasks. The preprocessing steps involve loading the dataset, limiting the
        vocabulary size, and padding the sequences to ensure uniform input
        length for the neural network.
        <br />
        First, we import the necessary libraries: numpy for numerical operations
        and tensorflow for building and training the neural network. We then
        load the IMDB dataset using imdb.load_data(), specifying a vocabulary
        size of 5000 words. This means that only the top 5000 most frequent
        words in the dataset will be considered, which helps in reducing the
        complexity and size of the input data.
        <br />
        Next, we use the pad_sequences function from tensorflow.keras.utils to
        pad the sequences. Padding ensures that all input sequences have the
        same length, which is crucial for processing the data in batches. Here,
        we set the maximum length of the sequences to 250. Any sequence shorter
        than this length will be padded with zeros, and any sequence longer will
        be truncated.
        <br />
        <br />
      </p>
      <h5>Step1b Simple RNN model</h5>

      <Code
        code={`
          # Import the necessary module from TensorFlow
          from tensorflow import keras
          # Define the input layer with a shape corresponding to the length of the padded sequences
          inputs = keras.layers.Input(shape=(250,))
          # Add an embedding layer to transform input integers into dense vectors of size 32
          x = keras.layers.Embedding(input_dim=vocab_size, output_dim=32, input_length=max_length)(inputs)
          # Add a SimpleRNN layer with 128 units and ReLU activation function
          x = keras.layers.SimpleRNN(128, activation='relu')(x)
          # Add a dense output layer with a sigmoid activation function for binary classification
          outputs = keras.layers.Dense(1, activation='sigmoid')(x)
          # Create the model by specifying the input and output layers
          model = keras.Model(inputs=inputs, outputs=outputs)
          # Print a summary of the model architecture
          model.summary()          
          `}
      />
      <p>
        In step b we construct a simple Recurrent Neural Network (RNN) model
        using TensorFlow’s Keras API. The model is designed for binary
        classification tasks, such as sentiment analysis on the IMDB dataset.
        The architecture includes an embedding layer, a simple RNN layer, and a
        dense output layer. First, we define the input layer with a shape of
        (250,), which corresponds to the length of the padded sequences. The
        embedding layer follows, which transforms the input integers into dense
        vectors of fixed size (32 in this case). This layer helps the model to
        learn useful representations of the words in the dataset.
        <br />
        Next, we add a SimpleRNN layer with 128 units and a ReLU activation
        function. The RNN layer processes the sequence data, capturing temporal
        dependencies and patterns within the sequences. The output from the RNN
        layer is then passed to a dense layer with a sigmoid activation
        function. This dense layer outputs a single value between 0 and 1,
        suitable for binary classification tasks. Finally, we create the model
        by specifying the input and output layers and print a summary of the
        model architecture.
        <br />
        <br />
      </p>
      <h5>Step1c: Model Conclusion</h5>
      <Code
        code={`
          model.compile(optimizer= 'adam',
                        loss = 'binary_crossentropy',
                        metrics = ['accuracy'])
          history = model.fit(x_train_pad, y_train, epochs=10, batch_size=32, validation_split=0.2 )
          model.save('rnn_imdb_sentiment.h5')          
          `}
      />
      <p>
        The model.compile method configures the model for training by specifying
        the optimizer, loss function, and evaluation metrics. Here, the Adam
        optimizer is used for its efficiency, and binary cross-entropy is chosen
        as the loss function, suitable for binary classification tasks. Accuracy
        is set as the evaluation metric to monitor the model’s performance. The
        model.fit method trains the model on the training data (x_train_pad and
        y_train) for 10 epochs with a batch size of 32. A validation split of
        20% is used to evaluate the model’s performance on unseen data during
        training. Finally, the trained model is saved to a file named
        ‘rnn_imdb_sentiment.h5’ using the model.save method, allowing for future
        use without retraining.
        <br />
        <br />
      </p>
      <h4>Step2 Deploy app using streamlit</h4>
      <Code
        code={`
            #StepB: Deploy AI model with streamlit "app.py"
            # Step 1: Import Libraries and Load the Model
            import numpy as np
            import tensorflow as tf
            from tensorflow.keras.datasets import imdb
            from tensorflow.keras.utils import pad_sequences
            from tensorflow.keras.models import load_model
            # Load the IMDB dataset word index
            word_index = imdb.get_word_index()
            reverse_word_index = {value: key for key, value in word_index.items()}
            # Load the pre-trained model with ReLU activation
            model = load_model('rnn_imdb_sentiment.h5')
            # Step 2: Helper Functions
            # Function to decode reviews
            def decode_review(encoded_review):
                # Convert encoded integers back to words
                return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])
            # Function to preprocess user input
            def preprocess_text(text):
                # Convert text to lowercase and split into words
                words = text.lower().split()
                # Encode words using the word index
                encoded_review = [word_index.get(word, 2) + 3 for word in words]
                # Pad the encoded review to a maximum length of 500
                padded_review = pad_sequences([encoded_review], maxlen=500)
                return padded_review
            # Step 3: Streamlit Interface
            # Import Streamlit library
            import streamlit as st
            # Streamlit app
            st.title('IMDB Movie Review Sentiment Analysis')
            st.write('Enter a movie review to classify it as positive or negative.')
            # User input
            user_input = st.text_area('Movie Review')
            if st.button('Classify'):
                # Preprocess the user input
                preprocessed_input = preprocess_text(user_input)
                # Make prediction
                prediction = model.predict(preprocessed_input)
                sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
                # Display the result
                st.write(f'Sentiment: {sentiment}')
                st.write(f'Prediction Score: {prediction[0][0]}')
            else:
                st.write('Please enter a movie review.')          
          `}
      />
      <p>
        First, we import essential libraries such as numpy for numerical
        operations, tensorflow for deep learning functionalities, and streamlit
        for creating the web interface. We then load the IMDB dataset’s word
        index and the pre-trained model.
        <br />
        We define two helper functions. The decode_review function converts
        encoded reviews back into readable text using the reverse word index.
        The preprocess_text function processes user input by converting text to
        lowercase, splitting it into words, encoding each word using the word
        index, and padding the sequence to a fixed length.
        <br />
        We set up a Streamlit interface that allows users to input movie reviews
        and classify them as positive or negative. The app displays a title and
        a text area for user input. When the “Classify” button is pressed, the
        user input is preprocessed, and the model predicts the sentiment. The
        result, either “Positive” or “Negative,” along with the prediction
        score, is displayed.
        <br />
        <br />
      </p>
    </div>
  );
};

export default Content015;
