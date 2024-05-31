import React from "react";
import "./sidebar.css"

const Sidebar = () => {
  return (
    <div className="sidebar">
      <button
        className="navbar-toggler"
        type="button"
        data-toggle="collapse"
        data-target="#navigation"
        aria-controls="navigation"
        aria-expanded="true"
        aria-label="Toggle navigation"
      >
        <span className="navbar-toggler-icon"></span>
      </button>
      <div
        className="sidebar-sticky d-flex flex-column  align-items-center"
        id="side-navbar"
      >
        <div className="web-brand d-flex flex-column align-items-center pt-5 pb-5">
          <div className="sidebar-logo  ">
            <i className="bi bi-cpu logo-icon"></i>
          </div>
          <div>
            <h1>AIBlog</h1>
          </div>
        </div>

        <ul className="nav flex-column gap-2">
          <li className="nav-item border rounded-pill">
            <a className="nav-link active fw-bold fs-3 " href="/">
              <i className="bi bi-house-fill"></i>
              Home
            </a>
          </li>
          <li className="nav-item border rounded-pill">
            <a className="nav-link active fw-bold fs-3" target="_blank" href="https://mohit-ram.github.io/portfolio/">
              <i className="bi bi-person-fill"></i>
              About
            </a>
          </li>
          <li className="nav-item border rounded-pill">
            <a className="nav-link active fw-bold fs-3 " target="_blank" href="https://mohit-ram.github.io/portfolio/">
              <i className="bi bi-file-person-fill"></i>
              Portfolio
            </a>
          </li>
          <li className="nav-item border rounded-pill ">
            <a className="nav-link active fw-bold fs-3" target="_blank" href="https://mohit-ram.github.io/portfolio/">
              <i className="bi bi-person-lines-fill"></i>
              Contact
            </a>
          </li>
        </ul>

        <div className="social-links   ">
          <a href="#">
            <i className="bi bi-facebook px-2"></i>
          </a>
          <a href="#">
            <i className="bi bi-twitter-x px-2"></i>
          </a>
          <a href="#">
            <i className="bi bi-git px-2"></i>
          </a>
        </div>
      </div>
    </div>
  );
};

export default Sidebar;
