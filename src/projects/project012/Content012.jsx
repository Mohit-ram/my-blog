import Code from "../../components/Code/Code.jsx";
import img01 from "./img01.png";
import img02 from "./img02.png";
import img03 from "./img03.png";
import img04 from "./img04.png";
import img05 from "./img05.png";
import img06 from "./img06.png";
import img07 from "./img07.png";
import img08 from "./img08.png";
import img09 from "./img09.png";
import img10 from "./img10.png";

const Content012 = () => {
  return (
    <div className="page container mt-5 mx-auto w-75 px-5 ">
      <h1 className="text-center">Pandas Bulit in Visualization</h1>
      <div className="text-center"></div>
      <p>
        In this project we explore various pandas data visualization function,
        which are easy to understand and can be directly called on dataframe to
        get a quick understanding of data. This much useful before going to the
        deep data analysis with advanced visualization techniques.
      </p>

      <h4> Plot Types </h4>
      <p>
        There are several plot types built-in to pandas, most of them
        statistical plots by nature:
        <br />
        df.plot.area
        <br />
        df.plot.barh
        <br />
        df.plot.density
        <br />
        df.plot.hist
        <br />
        df.plot.line
        <br />
        df.plot.scatter
        <br />
        df.plot.bar
        <br />
        df.plot.box
        <br />
        df.plot.hexbin
        <br />
        df.plot.kde
        <br />
        df.plot.pie
        <br />
        Below is the code with appropriate comments for various plots in pandas.
        Dataframes df1 and df2 are random generated float values to understand
        the plots.
        <br />
      </p>
      <Code
        code={`
          import numpy as np
          import pandas as pd
          # Plot an area chart with 40% transparency
          df2.plot.area(alpha=0.4)
          # Plot a stacked bar chart
          df2.plot.bar(stacked=True)
          # Plot a histogram of column 'A' with 50 bins
          df1['A'].plot.hist(bins=50)
          # Plot a line chart of column 'B' against the index, with specified figure size and line width
          df1.plot.line(x=df1.index, y='B', figsize=(12, 3), lw=1)
          # Plot a scatter plot of 'A' vs 'B', colored by 'C' using the 'coolwarm' colormap
          df1.plot.scatter(x='A', y='B', c='C', cmap='coolwarm')
          # Plot a scatter plot of 'A' vs 'B', with marker sizes proportional to 'C'
          df1.plot.scatter(x='A', y='B', s=df1['C']*200)
          # Plot a box plot
          df2.plot.box()
          # Create a DataFrame with 1000 rows of random numbers in columns 'a' and 'b'
          df = pd.DataFrame(np.random.randn(1000, 2), columns=['a', 'b'])
          # Plot a hexbin plot of 'a' vs 'b' with a grid size of 25 and the 'Oranges' colormap
          df.plot.hexbin(x='a', y='b', gridsize=25, cmap='Oranges')
          # Plot a Kernel Density Estimate (KDE) plot of column 'a'
          df2['a'].plot.kde()
          # Plot a density plot for all columns in df2
          df2.plot.density()
          
          `}
      />
      <h4>Output</h4>
      <div className="d-block text-center">
        <img
          src={img01}
          alt="result2"
          style={{ height: "300px", width: "300px" }}
        />
        <img
          src={img02}
          alt="result1"
          style={{ height: "300px", width: "300px" }}
        />
        <img
          src={img03}
          alt="result2"
          style={{ height: "300px", width: "300px" }}
        />
        <img
          src={img04}
          alt="result3"
          style={{ height: "300px", width: "300px" }}
        />
        <img
          src={img05}
          alt="result4"
          style={{ height: "300px", width: "300px" }}
        />
        <img
          src={img06}
          alt="result1"
          style={{ height: "300px", width: "300px" }}
        />
        <img
          src={img07}
          alt="result2"
          style={{ height: "300px", width: "300px" }}
        />
        <img
          src={img08}
          alt="result1"
          style={{ height: "300px", width: "300px" }}
        />
        <img
          src={img09}
          alt="result2"
          style={{ height: "300px", width: "300px" }}
        />
        <img
          src={img10}
          alt="result2"
          style={{ height: "300px", width: "400px" }}
        />
      </div>
    </div>
  );
};

export default Content012;
