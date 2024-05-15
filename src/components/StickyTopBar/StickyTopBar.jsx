

const StickyTopBar = () => {
  return (
    <div>
    <nav class="navbar  container-fluid  fixed-top">
      <a class="navbar-brand " href="./index.html">
        Go Home <i class="bi bi-house-up-fill"></i>
      </a>
      <a class="navbar-brand " href="./index.html">
        Git <i class="bi bi-house-up-fill"></i>
      </a>

      {/* <div class="collapse navbar-collapse" id="navbarCollapse">
        <ul class="navbar-nav mr-auto">
          <li class="nav-item active">
            <a class="nav-link" href="#">
              Git
            </a>
          </li>
        </ul>
      </div> */}
    </nav>
    </div>
  );
};

export default StickyTopBar;
