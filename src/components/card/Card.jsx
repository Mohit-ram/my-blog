import React from "react";

import "./card.css";

const Card = ({ image, title, info }) => {
  return (
    <>
      
        <div className="row">
          <div className="col" >
            <div className="card mb-4 shadow">
              <div className="card-body  d-lg-inline-flex gap-5 pt-2 pt-b3">
                <img src={image} className="card-img-top" alt="Project Thumbnail" />
                <div className="card-info">
                  <h5 className="card-title">{title}</h5>
                  <p className="card-text">
                    {info}
                  </p>
                  <a
                    href="#"
                    className="project-link m-0 texr-decoration-underline"
                  >
                    <p>Read More</p>
                  </a>
                </div>
              </div>
            </div>
          </div>
        
        
        
      </div>
    </>
  );
};

export default Card;