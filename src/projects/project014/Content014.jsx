import Code from "../../components/Code/Code.jsx";
import img01 from "./img01.png"

const Content014 = () => {
  return (
    <div className="page container mt-5 mx-auto w-75 px-5 ">
      <h1 className="text-center">Real Time Iris Prediction</h1>
      <div className="text-center"></div>
      <p>
        In this project we use multiple libraries including streamlit, sklearn
        to get various length and width values of iris flower form user and
        predict the species using pretrained model on iris_datset. And dispay in
        browser using streamlit interface.
      </p>
      <h4>app.py</h4>
      <Code
        code={`
          import streamlit as st
          import pandas as pd
          from sklearn.datasets import load_iris
          from sklearn.ensemble import RandomForestClassifier
          # Load the Iris dataset
          def load_data():
              iris = load_iris()
              df = pd.DataFrame(iris.data, columns=iris.feature_names)
              df['species'] = iris.target
              return df, iris.target_names
          df, target_names = load_data()
          features = df.drop("species", axis=1)
          # Initialize the Random Forest model
          model = RandomForestClassifier()
          model.fit(features, df['species'])
          # Sidebar input for feature values
          st.sidebar.title("Feature values")
          sepal_length = st.sidebar.slider("Sepal Length", float(df['sepal length (cm)'].min()), float(df['sepal length (cm)'].max()))
          sepal_width = st.sidebar.slider("Sepal Width", float(df['sepal width (cm)'].min()), float(df['sepal width (cm)'].max()))
          petal_length = st.sidebar.slider("Petal Length", float(df['petal length (cm)'].min()), float(df['petal length (cm)'].max()))
          petal_width = st.sidebar.slider("Petal Width", float(df['petal width (cm)'].min()), float(df['petal width (cm)'].max()))
          # Create input data for prediction
          input_data = [[sepal_length, sepal_width, petal_length, petal_width]]
          # Make the prediction
          prediction = model.predict(input_data)
          predicted_species = target_names[prediction[0]]
          # Display the result
          st.write(f"The predicted species based on given input features is: {predicted_species}")

          `}
      />
      <p>
        We begin by importing necessary libraries: streamlit, pandas, and
        RandomForestClassifier from sklearn. The load_iris dataset contains
        information about iris flowers, including features like sepal length,
        sepal width, petal length, and petal width. We create a DataFrame (df)
        to hold the iris data and add a new column called “species” to store the
        target labels (setosa, versicolor, or virginica). We initialize a
        RandomForestClassifier model, which is an ensemble learning method based
        on decision trees. The model is trained using the features (sepal
        length, sepal width, petal length, and petal width) and the
        corresponding species labels.
        <br />
        The Streamlit app displays a sidebar with sliders for user input. Users
        can adjust the sepal length, sepal width, petal length, and petal width
        to predict the species. The user’s input values are collected and stored
        in input_data. The trained model predicts the species based on these
        input features. The predicted species label is retrieved from
        target_names. The predicted species is shown to the user.
        <br />
        <br />
      </p>
      <p> To run the app, execute streamlit run app.py in terminal</p>
      <div className=" text-center">
        <img
          src={img01}
          alt="result1"
          style={{ height: "400px", width: "640px" }}
        />
      </div>
    </div>
  );
};

export default Content014;
