import React from "react";
import Sidebar from "../sidebar/Sidebar";
import SidePage from "../SidePage/SidePage";
import "./homepage.css";

const Page = () => {
  return (
    <>
      
      <div className="container-fluid">
        <Sidebar />
        <SidePage />
        
      </div>
    </>
  );
};

export default Page;
