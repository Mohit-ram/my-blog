import Code from "../../components/Code/Code.jsx";
import mainImg from "./mainImg002.jpg";

const Content002 = () => {
  return (
    <div className="page container mt-5 mx-auto w-75 px-5 ">
      <h1 className="text-center">Project Title</h1>
      <div className="text-center">
        <img src={mainImg} className="h-50 w-50"></img>
      </div>
      <p>
        Project intro
      </p>
      <h4></h4>
      <Code
        code={`
          
          `}
      />
      <p>
        <br />
        <br />
      </p>
    </div>
  );
};

export default Content002;
