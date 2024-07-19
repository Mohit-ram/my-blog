import React from 'react'
import "./catalogbar.css"

const CatalogBar = ({showAllProjects, showBeginnerProjects, showInterProjects, showOpenCVProjects, showLLMProjects}) => {
  return (
    <div className="container ">
    <header className="d-flex project-catalogbar flex-wrap align-items-center justify-content-center justify-content-md-between py-3 mb-4 border-bottom border-3">
      <h5>Catalog</h5>
      <ul className="nav col-12 col-md-auto mb-2 justify-content-center mb-md-0">
        <li>
          {/* <a
            href="#"
            className="nav-link  catalog-link px-2 link-secondary"
          >
            New
          </a> */}
        </li>
        <li>
          <a href="#" onClick={() => showAllProjects()} className="nav-link  catalog-link px-2">
            All
          </a>
        </li>
        <li>
          <a href="#" onClick={() => showBeginnerProjects()}className="nav-link catalog-link px-2">
            Beginner
          </a>
        </li>
        <li>
          <a href="#" onClick={() => showInterProjects()}className="nav-link  catalog-link px-2">
            Intermediate
          </a>
        </li>
        <li>
          <a href="#" onClick={() => showOpenCVProjects()}className="nav-link  catalog-link px-2">
            OpenCV
          </a>
        </li>
        <li>
          <a href="#" onClick={() => showLLMProjects()}className="nav-link catalog-link px-2">
            LLM
          </a>
        </li>
      </ul>
    </header>
  </div>
  )
}

export default CatalogBar