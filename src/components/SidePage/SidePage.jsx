import React from 'react'
import SearchPage from "../SearchPage/SearchPage";
import "./sidepage.css"

const SidePage = () => {
  return (
    <div  className="flex-column content-area ">
          <div className="jumbotron">
            <div className="p-5 text-center bg-body-tertiary">
              <div className="container py-5">
                <h1 className="text-body-emphasis">Full-width jumbotron</h1>
                <p className="col-lg-8 mx-auto lead">
                  This takes the basic jumbotron above and makes its background
                  edge-to-edge with a <code>.container</code> inside to align
                  content. Similar to above, it's been recreated with built-in
                  grid and utility classes.
                </p>
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