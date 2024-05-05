import React, {useState} from 'react'
import Posts_test from './posts_test.jsx'
import projects from './project-catalog.js'
import Pagination from './Pagination.jsx';


function createpost(project){
    return (<Posts_test prop={project} />);

}
const Posts = () => {

    const [currentPage, setCurrentPage] = useState(1);
    const [postsPerPage] = useState(5);
    
    const totalPages = Math.ceil(projects.length / postsPerPage);
    
    const indexOfLastPost = currentPage *  postsPerPage;
    console.log(totalPages)
    const indexOfFirstPost = indexOfLastPost - postsPerPage;
    const currentPosts = projects.slice(indexOfFirstPost, indexOfLastPost);
    const nextPage = pageNumber => {
      if (pageNumber == totalPages){
      return setCurrentPage(pageNumber)}
      else{ return setCurrentPage(pageNumber+1)}
    };
    const previousPage = pageNumber => {
      if (pageNumber == 1){
      return setCurrentPage(pageNumber)}
      else{ return setCurrentPage(pageNumber-1)}
    };
    return (
    <div>
        <ul>
        {currentPosts.map(createpost)}
        </ul>
        
        <Pagination
        pageNumber= {currentPage}
        previousPage={previousPage}
        nextPage={nextPage}
      />
    </div>
  )
}

export default Posts