import React from "react";
import "./pagination.css"

const Pagination = ({ pageNumber, previousPage, nextPage }) => {
  return (
    <div>
      <nav aria-label="Page navigation example">
        <ul className="pagination pagination-lg justify-content-center gap-3">
          <li className="page-item border rounded-pill">
            <a
              className="page-link "
              href="#"
              aria-label="Previous"
              onClick={() => previousPage(pageNumber)}
            >
              <span aria-hidden="true">&laquo;</span>
            </a>
          </li>
          <li className="page-item ">
            <a className="page-link" href="!#">
              {pageNumber}
            </a>
          </li>

          <li className="page-item">
            <a
              className="page-link"
              href="#"
              aria-label="Next"
              onClick={() => nextPage(pageNumber)}
            >
              <span aria-hidden="true">&raquo;</span>
            </a>
          </li>
        </ul>
      </nav>
    </div>
  );
};

export default Pagination;
