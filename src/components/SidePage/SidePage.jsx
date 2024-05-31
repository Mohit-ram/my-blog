import React from "react";
import SearchPage from "../SearchPage/SearchPage";
import "./sidepage.css";
import bgImage from "../../assets/images/bgimg.jpg";

const SidePage = () => {
  return (
    <div className=" content-area ">
      <div
        className="jumbo-container py-5 "
        style={{ backgroundImage: `url(${bgImage})`, backgroundSize: "cover" }}
      >
        
      </div>
      <div className="container text-center">
            <h4 className="text-body-emphasis">
              Discover and learn AI in exciting way with new techniques and more
              with me!!.
            </h4>
          </div>
     
      <div className="project-navigation">
        <SearchPage />
      </div>
    </div>
  );
};

export default SidePage;
