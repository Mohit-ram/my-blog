import React, { useState } from "react";
import allProjects, {
  onlyCatA,
  onlyCatB,
  onlyCatC,
  onlyCatD,
} from "../../projects/project-catalog.js";
import Pagination from "../Pagination/Pagination.jsx";
import Card from "../card/Card.jsx";
import CatalogBar from "../catalog_bar/CatalogBar.jsx";
// import image from "/src/assets/images/img1.jpg";

function createcard(project) {
  return (
    <Card
      key={project.Id}
      number = {project.number}
      image={project.imgPath}
      title={project.title}
      info={project.info}
    />
  );
  
}

const SearchPage = () => {
  const [projects, setProjects] = useState(allProjects);
  const [currentPage, setCurrentPage] = useState(1);
  const [postsPerPage] = useState(5);

  console.log(projects[0].Id);
  const showAllProjects = () => {
    setProjects(projects);
    setCurrentPage(1);
  };
  const showBeginnerProjects = () => {
    setProjects(onlyCatA());
    setCurrentPage(1);
  };
  const showInterProjects = () => {
    setProjects(onlyCatB());
    setCurrentPage(1);
  };
  const showOpenCVProjects = () => {
    setProjects(onlyCatC());
    setCurrentPage(1);
  };
  const showLLMProjects = () => {
    setProjects(onlyCatD());
    setCurrentPage(1);
  };
  const totalPages = Math.ceil(projects.length / postsPerPage);

  const indexOfLastPost = currentPage * postsPerPage;
  console.log(totalPages);
  const indexOfFirstPost = indexOfLastPost - postsPerPage;
  const currentPosts = projects.slice(indexOfFirstPost, indexOfLastPost);
  const nextPage = (pageNumber) => {
    if (pageNumber == totalPages) {
      return setCurrentPage(pageNumber);
    } else {
      return setCurrentPage(pageNumber + 1);
    }
  };
  const previousPage = (pageNumber) => {
    if (pageNumber == 1) {
      return setCurrentPage(pageNumber);
    } else {
      return setCurrentPage(pageNumber - 1);
    }
  };

  return (
    <>
      <CatalogBar
        showAllProjects={showAllProjects}
        showBeginnerProjects={showBeginnerProjects}
        showInterProjects={showInterProjects}
        showOpenCVProjects={showOpenCVProjects}
        showLLMProjects={showLLMProjects}
      />

      <div className="project-cards">
        {currentPosts.map(createcard)}
      </div>

      <Pagination
        pageNumber={currentPage}
        previousPage={previousPage}
        nextPage={nextPage}
      />
    </>
  );
};

export default SearchPage;
