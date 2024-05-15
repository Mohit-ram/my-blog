const step01 = {
    'codesnippet': `# Step 1: Data Preprocessing
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
    # The output will show the number of samples for each species.`,
};

const content01 = (
    `<h1>heading </h1>
    <p>para</p>`
)
export default step01;
export {content01}
