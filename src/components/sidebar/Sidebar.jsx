import React from "react";
import "./sidebar.css";

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
        <div className="web-brand d-flex flex-column justify-content-center align-items-center pt-5 pb-5">
          <div className="sidebar-logo  ">
            <i className="bi bi-cpu logo-icon"></i>
          </div>
          <div>
            <h1>Growing AI</h1>
            
          </div>
        </div>
      </div>
    </div>
  );
};

export default Sidebar;
