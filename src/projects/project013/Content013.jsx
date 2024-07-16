import Code from "../../components/Code/Code.jsx";

const Content013 = () => {
  return (
    <div className="page container mt-5 mx-auto w-75 px-5 ">
      <h1 className="text-center">Quick guide to numpy</h1>
      <div className="text-center">
        
      </div>
      <p>
        This quick read is to reference most widely used numpy concepts and
        operations. The code snippets provided below has various operation on
        arrays and can be used a quick guide.
      </p>
      <h4>Basic Array Operations</h4>
      <Code
        code={`
          # Creating a 1D array
          arr1 = np.array([1, 2, 3, 4, 5])
          # Creating a 2D array
          arr2 = np.array([[1, 2, 3], [4, 5, 6]])

          arr = np.array([1, 2, 3, 4, 5])

          # Indexing and Slicing
          # Accessing an element
          element = arr[2]  # Output: 3

          arr = np.array([10, 20, 30, 40, 50])
          # Access elements at indices 1, 3, and 4
          result = arr[[1, 3, 4]]  # Output: [20, 40, 50]

          # Slicing an array
          slice_arr = arr[1:4]  # Output: [2, 3, 4]

          #Reshaping Arrays
          arr = np.array([[1, 2, 3], [4, 5, 6]])

          # Reshaping the array to 3x2
          reshaped_arr = arr.reshape(3, 2)
          `}
      />
      <h4>Mathematical Operations</h4>
      <Code
        code={`
          # Adding two arrays
          arr1 = np.array([1, 2, 3])
          arr2 = np.array([4, 5, 6])
          result = arr1 + arr2  # Output: [5, 7, 9]

          # Multiplying two arrays
          result = arr1 * arr2  # Output: [4, 10, 18]

          arr = np.array([1, 2, 3, 4, 5])

          # Calculating the sum of all elements
          sum_arr = np.sum(arr)  # Output: 15

          # Calculating the mean of all elements
          mean_arr = np.mean(arr)  # Output: 3.0

          # Calculating the standard deviation
          std_arr = np.std(arr)  # Output: 1.4142135623730951


          `}
      />
      <h4>Random numbers</h4>
      <Code
        code={`
          # Generate an array of random numbers
          random_arr = np.random.rand(3, 3)

          # Generate random integers
          random_ints = np.random.randint(0, 10, size=(3, 3))

          # Set a random seed for reproducibility
          np.random.seed(42)
          random_arr = np.random.rand(3, 3)

          `}
      />
      <h4>Stacking arrays</h4>
      <Code
        code={`
          arr = np.array([[1, 2], [3, 4], [5, 6]])

          # Split the array into three parts
          split_arr = np.split(arr, 3)

          # Stack arrays vertically
          vstack_arr = np.vstack((arr, arr))

          # Stack arrays horizontally
          hstack_arr = np.hstack((arr, arr))
        
          `}
      />
      <h4>2D and 3D arrays</h4>
      <Code
        code={`
          # Creating an array of zeros
          zeros = np.zeros((3, 3))

          # Creating an array of ones
          ones = np.ones((2, 4))

          # Creating an array with a range of values
          range_array = np.arange(0, 10, 2)  # Output: [0, 2, 4, 6, 8]

          # Creating an array with random values
          random_array = np.random.rand(3, 3)


          #Dot product
          arr1 = np.array([1, 2, 3])
          arr2 = np.array([4, 5, 6])

          dot_product = np.dot(arr1, arr2)  # Output: 32

          # Creating matrices
          matrix1 = np.array([[1, 2], [3, 4]])
          matrix2 = np.array([[5, 6], [7, 8]])

          # Matrix multiplication
          result = np.matmul(matrix1, matrix2)  # Output: [[19, 22], [43, 50]]

          # Eigenvalues and eigenvectors
          eigenvalues, eigenvectors = np.linalg.eig(matrix1)        
          `}
      />
     
    </div>
  );
};

export default Content013;
