import Code from "../../components/Code/Code.jsx";
import mainImg from "./mainImg001.jpg";

const Content001 = () => {
  return (
    <div className="page container mt-5 mx-auto w-75 px-5 ">
      <h1 className="text-center">Iris Species Detection</h1>
      <div className="text-center">
        <img src={mainImg} className="h-50 w-50"></img>
      </div>
      <p>
        A simple classification program in Python to predict the species of an
        Iris flower using various classification algorithms. The objective of
        these projects is to build various predictive models capable of distinguishing
        between the three species of Iris flowers—setosa, versicolor, and
        virginica—based on the physical dimensions of their petals and sepals.
        By applying machine learning techniques.
      </p>
      <h4>Step1: Data Preprocessing - Read and Ready the Data.</h4>
      <Code
        code={`# Step 1: Data Preprocessing
          #Step1A: Read the data
          # Import necessary libraries
          import pandas as pd
          import numpy as np    
          # Read the data from the CSV file
          data = pd.read_csv('iris.csv')    
          # Explore the dataset
          # Check the distribution of species in the dataset
          species_distribution = data['species'].value_counts()
          print(species_distribution)    
          # The output will show the number of samples for each species.`}
      />
      <p>
        Reading and Understanding the Data: The first step in data preprocessing
        is to read the dataset from its source. This is typically done using
        libraries like pandas, which is a powerful Python library for data
        analysis. In the provided code, the pd.read_csv('iris.csv') function is
        used to read the Iris dataset, which is a commonly used dataset in
        machine learning and statistics. This dataset includes various
        measurements of iris flowers and their corresponding species.
        <br />
        <br />
        Once the data is loaded into a pandas DataFrame, it’s essential to
        understand its structure. This involves checking the size of the
        dataset, the number of samples (rows), the number of features (columns),
        the types of features (numerical, categorical), and identifying any
        missing values. The data['species'].value_counts() command counts the
        number of occurrences of each species in the dataset, providing insight
        into the distribution of samples.
        <br />
        <br />
        Data Cleaning and Preprocessing: After gaining an initial understanding
        of the dataset, the next step is to clean and preprocess the data. This
        involves handling missing values, which can be done through methods such
        as simple imputation, where missing values are replaced with statistical
        measures like mean, median, or mode.
        <br />
        <br />
        It’s also important to examine the distribution of samples. If the
        dataset is imbalanced (i.e., some classes are overrepresented compared
        to others), it may require resampling techniques to ensure that the
        model trained on this data does not become biased towards the more
        frequent classes.
        
      </p>
      <h4>Step1B: Visualizing Data for Pattern Recognition</h4>
      <Code
        code={`
      # Visualise the Data for Pattern Recognition
      # Import visualization libraries
      import seaborn as sns
      import matplotlib.pyplot as plt
      # Create a pairplot of the dataset
      # This will plot pairwise relationships in the dataset and color the points by species
      pair_plot = sns.pairplot(data=data, hue='species')
      # Display the pairplot
      plt.show()
      # The pairplot helps in identifying patterns and relationships between features.
      # It is also useful for spotting important features for feature engineering.

      `}
      />
      <p>
        Data visualization is a crucial step in data analysis as it allows for
        the identification of patterns, trends, and correlations that might not
        be evident from raw data alone. In the context of the Iris dataset,
        visualizing the data can help in understanding how the features relate
        to each other and how they are distributed across different species.
        <br />
        <br />
        The code snippet provided uses the seaborn library, which is built on
        top of matplotlib and provides a high-level interface for drawing
        attractive and informative statistical graphics. The sns.pairplot
        function creates a grid of Axes such that each variable in the dataset
        will be shared across the y-axes across a single row and the x-axes
        across a single column. The hue parameter is used to color the data
        points by the ‘species’ column, which helps in distinguishing the
        different species from each other. By visualizing the data, one can
        identify which features are important and how they can be transformed or
        combined to improve the model’s performance.For instance, if the
        pairplot shows that certain features separate the species well, those
        features might be particularly useful for classification tasks.
        Conversely, if two features are highly correlated, we might consider
        removing one to reduce redundancy.
        
      </p>
      <h4>Step1C: Feature Engineering(Data Preparation)</h4>
      <Code
        code={`
              # Ready the Data for Machine Learning
              # Import necessary modules from scikit-learn
              from sklearn.preprocessing import LabelEncoder
              from sklearn.model_selection import train_test_split
              # Convert the 'species' column into an array and encode it numerically
              y = np.array(data['species'])
              y_encoded = LabelEncoder().fit_transform(y)
              # Drop the 'species' column from the data and convert the rest into an array for feature set
              x = np.array(data.drop('species', axis=1))
              # Split the dataset into training and testing sets
              # Here, 15% of the data will be used for testing and the rest for training
              x_train, x_test, y_train, y_test = train_test_split(x, y_encoded, test_size=0.15)
              # Print the shapes of the resulting arrays to ensure everything is as expected
              shapes = [v.shape for v in [x_train, x_test, y_train, y_test]]
              print(shapes)
          `}
      />
      <p>
        Feature engineering is a critical process in machine learning where we
        transform raw data into features that better represent the underlying
        problem to the predictive models, resulting in improved model accuracy
        on unseen data.
        <br />
        <br />
        In above code, feature engineering involves encoding categorical
        variables into a format that can be provided to machine learning
        algorithms to do a better job in prediction. The LabelEncoder from
        sklearn.preprocessing is a utility class to help normalize labels such
        that they contain only values between 0 and n_classes-1. This is useful
        for converting categorical data, or text data, into numbers, which the
        machine learning model can understand. The LabelEncoder is applied to
        the ‘species’ column, which is a categorical feature representing the
        species of iris flowers. By encoding this column, we convert the species
        names into a numerical format that can be understood by machine learning
        algorithms.
        <br />
        <br />
        Species: [Iris Setosa,Iris Versicolour,Iris Virginica] -[0,1,2] The
        train_test_split function is then used to divide the dataset into
        training and testing sets. This is a common practice in machine learning
        to evaluate the performance of a model. The test_size=0.15 parameter
        indicates that 15% of the data will be set aside for testing the model,
        and the remaining 85% will be used for training.
      </p>
      <h4>Step2: Model selection and hyperparameters</h4>
      <Code
        code={`
              # Step 2: Choosing Models and Hyperparameters
              # Import machine learning models from scikit-learn
              from sklearn.neighbors import KNeighborsClassifier
              from sklearn.svm import SVC
              from sklearn.ensemble import HistGradientBoostingClassifier
              # Define hyperparameter grids for each model
              HGB_params_grid = {
                  'max_depth': [5, 10, 15, 20, 25, 30, None],
                  'learning_rate': [0.05, 0.1, 0.15]
              }
              KNN_params_grid = {
                  'n_neighbors': range(1, 21, 2),
                  'weights': ['uniform', 'distance'],
                  'metric': ['euclidean', 'manhattan', 'minkowski']
              }
              SVC_params_grid = {
                  'kernel': ['poly', 'rbf', 'sigmoid'],
                  'C': [50, 10, 1.0, 0.1, 0.01],
                  'gamma': ['scale']
              }
              # Organize models and their hyperparameters in a dictionary
              models = {
                  'HGB': {'model': HistGradientBoostingClassifier(), 'params': HGB_params_grid},
                  'KNN': {'model': KNeighborsClassifier(), 'params': KNN_params_grid},
                  'SVC': {'model': SVC(), 'params': SVC_params_grid}
              }
          
          `}
      />
      <p>
        In the process of building a machine learning model, selecting the
        appropriate algorithms and tuning their hyperparameters are crucial
        steps that can significantly impact the performance of the model.
        <br />
        <br />
        Model Selection: The code snippet outlines the use of three different
        machine learning models: HistGradientBoostingClassifier,
        KNeighborsClassifier, and SVC (Support Vector Classifier). Each of these
        models has its strengths and is suitable for different types of data and
        problems.
        <br />
        <br />
        HistGradientBoostingClassifier: This is a gradient boosting model for
        classification tasks which is effective for datasets with a large number
        of features and instances. It is robust to outliers and can handle
        missing values natively. KNeighborsClassifier: This model implements the
        k-nearest neighbors voting algorithm, which is simple yet effective for
        small to medium-sized datasets. It is a non-parametric method used for
        classification and regression. SVC: The Support Vector Classifier is a
        powerful algorithm that works well for both linear and non-linear
        boundaries. It is particularly useful for datasets where the classes are
        not linearly separable. Hyperparameter Tuning: Hyperparameters are the
        configuration settings used to structure the machine learning model.
        Proper tuning of these parameters can lead to better model accuracy.
        <br />
        <br />
        HGB_params_grid: This dictionary contains hyperparameters for the
        HistGradientBoostingClassifier, such as max_depth, which determines the
        maximum depth of the individual estimators, and learning_rate, which
        affects the contribution of each tree in the ensemble. KNN_params_grid:
        For the KNeighborsClassifier, the parameters include n_neighbors, which
        specifies the number of neighbors to use for predictions, weights, which
        determines the weight function used in prediction, and metric, which
        defines the distance metric for the tree. SVC_params_grid: The
        hyperparameters for the SVC include kernel, which specifies the kernel
        type to be used in the algorithm, C, which is the regularization
        parameter, and gamma, which defines the influence of a single training
        example. Models Grid: The models dictionary is a structured way to
        organize the models and their corresponding hyperparameters. This setup
        is particularly useful when performing grid search for hyperparameter
        tuning, as it allows for systematic exploration of the hyperparameter
        space for each model.
        
      </p>

      <h4>Step3 & Step4: Model evaluation and Hypertuning</h4>
      <Code
        code={`
            # Step 3 and Step 4: Model Evaluation and Hyperparameter Tuning
            # Import RandomizedSearchCV for hyperparameter tuning
            from sklearn.model_selection import RandomizedSearchCV
            # Define a custom evaluation function
            def custom_eval(models, x_train, y_train, x_test, y_test):
                # Initialize dictionaries to store scores and best parameters
                models_scores = {}  # Example: 'KNN': { score_metric1: [scores for five CV tests], score_metric2: ... }
                model_best_params = {}            
                # Iterate over the models and their parameters
                for name, model in models.items():
                    print(name)
                    print(model['model'])                    
                    # Set up RandomizedSearchCV with the model and its hyperparameter grid
                    custom_model = RandomizedSearchCV(estimator=model['model'],
                                                      param_distributions=model['params'],
                                                      n_iter=5, cv=5, verbose=2, n_jobs=1)
                    # Fit the model to the training data
                    custom_model.fit(x_train, y_train)
                    # Store the best parameters and the score of the best model
                    model_best_params[name] = custom_model.best_params_
                    models_scores[name] = custom_model.score(x_test, y_test)
                # Return the best parameters and scores for each model
                return model_best_params, models_scores          
          `}
      />
      <p>
        Model Evaluation: Model evaluation is the phase where the performance of
        different machine learning models is assessed. This is typically done by
        applying the models to a set of data that they have not seen before,
        known as the test set. The performance is measured using various
        metrics, such as accuracy, precision, recall, or F1 score, depending on
        the nature of the problem.
        <br />
        <br />
        Hyperparameter Tuning: Hyperparameter tuning involves adjusting the
        parameters of the model that are not learned from the data but are set
        prior to the training process. These parameters can significantly
        influence the performance of the model. The RandomizedSearchCV function
        from scikit-learn is used for hyperparameter tuning. It randomly samples
        from the given hyperparameter space and performs cross-validation to
        evaluate the models.
        <br />
        <br />
        Custom Evaluation Function: The custom_eval function defined in the code
        takes a dictionary of models and their associated hyperparameters, along
        with training and testing data. It then iterates over each model,
        performs randomized search cross-validation, and fits the model to the
        training data. The best hyperparameters for each model are stored, and
        the performance score of the best model is calculated on the test data.
       
      </p>
      <h4>Step5: Fitting the best Model</h4>
      <Code
        code={`
              # Step 4: Fitting the Best Model from Best Parameters
              # Retrieve the best parameters and scores from the custom evaluation
              params, scores = custom_eval(models, x_train, y_train, x_test, y_test)
              # Instantiate the model with the best parameters
              model = KNeighborsClassifier(weights='distance', n_neighbors=11, metric='minkowski')
              # Import cross_val_score for cross-validation
              from sklearn.model_selection import cross_val_score
              # Perform cross-validation to estimate the model's performance
              cross_validation_scores = cross_val_score(model, x_train, y_train, cv=5)
              # Calculate the mean of the cross-validation scores
              average_score = cross_validation_scores.mean()
              print(f"Average Cross-Validation Score: {average_score}")
              # Fit the model to the training data
              model.fit(x_train, y_train)
              # Evaluate the model's performance on the test data
              evaluation_score = model.score(x_test, y_test)
              print(f"Evaluation Score on Test Data: {evaluation_score}")
          
          `}
      />
      <p>
        The final step in the machine learning workflow is to fit the best model
        identified through hyperparameter tuning to the training data and
        evaluate its performance. BestModel: After identifying the best
        hyperparameters for each model using the custom_eval function, the next
        step is to instantiate the model with these parameters. In the code
        snippet, the KNeighborsClassifier is chosen with the hyperparameters
        weights='distance', n_neighbors=11, and metric='minkowski'. These
        parameters were likely determined to be the best combination during the
        hyperparameter tuning process.
        <br />
        <br />
        Cross-Validation: Before finalizing the model, it’s common practice to
        perform cross-validation to estimate its performance. The
        cross_val_score function is used to assess the model’s accuracy by
        dividing the training data into cv=5 folds, fitting the model on four
        folds, and validating it on the fifth fold. This process is repeated
        five times, each time with a different fold used as the validation set.
        The score.mean() function calculates the average score across all
        cross-validation folds, providing an estimate of the model’s
        performance.
        <br />
        <br />
        Training and Evaluation: Finally, the model is fitted to the entire
        training dataset using the model.fit(x_train, y_train) method. After
        training, the model’s performance is evaluated on the test set with the
        model.score(x_test, y_test) method, which returns the accuracy of the
        model.
        
      </p>

      <h4>Step6 :Saving and Loading the Model</h4>
      <Code
        code={`
            #import joblib
            # Save the model
            joblib.dump(model, 'clf.pkl')
            # Load the model
            loaded_model = joblib.load('clf.pkl')
            # Predict using the loaded model
            prediction = loaded_model.predict([x[149]])
          `}
      />
      <p>
        Save the trained KNeighborsClassifier model to a file named ‘clf.pkl’
        using joblib.dump. Load the model from ‘clf.pkl’ when needed using
        joblib.load. Make predictions using the loaded model with predict
        method.
        <br />
        <br />
      </p>
    </div>
  );
};

export default Content001;
