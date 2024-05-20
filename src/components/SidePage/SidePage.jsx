import React from 'react'
import SearchPage from "../SearchPage/SearchPage";
import "./sidepage.css"
import bgImg from "../../assets/images/bgimg.jpg"

const SidePage = () => {
  return (
    <div  className=" content-area ">
          <div className="jumbotron py-5 " >            
            <div className="p-5 text-center " >
              <div className="container py-5">
                <h3 className="text-body-emphasis">
                
                  Discover and learn AI in exciting way with new techniques and more with me!!.
                </h3>
              </div>
            </div>
          </div>

          <div className="project-navigation">
            

            <SearchPage />

            
          </div>
        </div>
  )
}

export default SidePage