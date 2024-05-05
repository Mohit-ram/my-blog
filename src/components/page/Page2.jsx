import React from "react";
import Sidebar from "../sidebar/Sidebar";
import SearchPage from "../SearchPage/SearchPage";
import "./page.css";

const Page = () => {
  return (
    <>
      {/* <div className="container-fluid">
        <div className="row">
            <div className="col-sm-2 col-md-4   vh-100">
                <Sidebar />
            </div>
            <div className="col-sm-10 col-md-8  mainpage bg-secondary vh-100">mainpage</div>
        </div>
        
  </div> */}
      <div className="container-fluid">
        <div className="sidebar">
          <button
            class="navbar-toggler"
            type="button"
            data-toggle="collapse"
            data-target="#navigation"
            aria-controls="navigation"
            aria-expanded="true"
            aria-label="Toggle navigation"
          >
            <span class="navbar-toggler-icon"></span>
          </button>
          <div
            className="sidebar-sticky d-flex flex-column  align-items-center"
            id="side-navbar"
          >
            <div className="web-brand d-flex flex-column align-items-center pt-5 pb-5">
              <div className="sidebar-logo  ">
                <i class="bi bi-cpu logo-icon"></i>
              </div>
              <div>
                <h1>AIBlog</h1>
              </div>
            </div>

            <ul className="nav flex-column gap-2">
              <li className="nav-item border rounded-pill">
                <a className="nav-link active fw-bold fs-3 " href="#">
                  <i class="bi bi-house-fill"></i>
                  Home
                </a>
              </li>
              <li className="nav-item border rounded-pill">
                <a className="nav-link active fw-bold fs-3" href="#">
                  <i class="bi bi-person-fill"></i>
                  About
                </a>
              </li>
              <li className="nav-item border rounded-pill">
                <a className="nav-link active fw-bold fs-3 " href="#">
                  <i class="bi bi-file-person-fill"></i>
                  Portfolio
                </a>
              </li>
              <li className="nav-item border rounded-pill ">
                <a className="nav-link active fw-bold fs-3" href="#">
                  <i class="bi bi-person-lines-fill"></i>
                  Contact
                </a>
              </li>
            </ul>

            <div className="social-links   ">
              <a href="#">
                <i class="bi bi-facebook px-2"></i>
              </a>
              <a href="#">
                <i class="bi bi-twitter-x px-2"></i>
              </a>
              <a href="#">
                <i class="bi bi-git px-2"></i>
              </a>
            </div>
          </div>
        </div>

        <div role="main" className="flex-column content-area ">
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
            <div className="container ">
              <header class="d-flex project-catalogbar flex-wrap align-items-center justify-content-center justify-content-md-between py-3 mb-4 border-bottom border-3">
                <h5>Catalog</h5>
                <ul class="nav col-12 col-md-auto mb-2 justify-content-center mb-md-0">
                  <li>
                    <a
                      href="#"
                      class="nav-link  catalog-link px-2 link-secondary"
                    >
                      New
                    </a>
                  </li>
                  <li>
                    <a href="#" class="nav-link  catalog-link px-2">
                      All
                    </a>
                  </li>
                  <li>
                    <a href="#" class="nav-link catalog-link px-2">
                      Beginner
                    </a>
                  </li>
                  <li>
                    <a href="#" class="nav-link  catalog-link px-2">
                      Intermediate
                    </a>
                  </li>
                  <li>
                    <a href="#" class="nav-link  catalog-link px-2">
                      OpenCV
                    </a>
                  </li>
                  <li>
                    <a href="#" class="nav-link catalog-link px-2">
                      LLM
                    </a>
                  </li>
                </ul>
              </header>
            </div>

            <SearchPage />

            
          </div>
        </div>
      </div>
    </>
  );
};

export default Page;
