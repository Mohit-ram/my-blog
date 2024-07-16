import Code from "../../components/Code/Code.jsx";

const Content011 = () => {
  return (
    <div className="page container mt-5 mx-auto w-75 px-5 ">
      <h1 className="text-center">SQL Advanced Queries</h1>
      <div className="text-center"></div>
      <p>
        In this project we explore more adavnced queries for deep data analysis.
        we use the same dataset as in our sql basic queries project
      </p>
      <h4>Query 1</h4>
      <p>
        1. Write a query that returns the following:emp_id,first_name,last_name,
        position_title, salary and a column that returns the average salary for
        every job_position. Order the results by the emp_id.
        <br />
        2.How many people earn less than there avg_position_salary?
        <br />
        3.Write a query that answers that question.Ideally, the output just
        shows that number directly.
      </p>
      <Code
        code={`
          SELECT
          emp_id,
          first_name,
          last_name,
          position_title,
          salary,
          ROUND(AVG(salary) OVER(PARTITION BY position_title),2) as avg_position_sal
          FROM employees
          ORDER BY 1;

          SELECT
          COUNT(*)
          FROM (
          SELECT 
          emp_id,
          salary,
          ROUND(AVG(salary) OVER(PARTITION BY position_title),2) as avg_pos_sal
          FROM employees) a
          WHERE salary < avg_pos_sal;
          
          `}
      />
      <h4>Query 2</h4>
      <p>
        Write a query that returns a running total of the salary development
        ordered by the start_date.
      </p>
      <Code
        code={`
          SELECT 
          emp_id,
          salary,
          start_date,
          SUM(salary) OVER(ORDER BY start_date) as salary_totals
          FROM employees;
          
          `}
      />
      <h4>Query 3 </h4>
      <p>
        Create the same running total but now also consider the fact that people
        were leaving the company.
      </p>
      <Code
        code={`
          SELECT 
          start_date,
          SUM(salary) OVER(ORDER BY start_date)
          FROM (
          SELECT 
          emp_id,
          salary,
          start_date
          FROM employees
          UNION 
          SELECT 
          emp_id,
          -salary,
          end_date
          FROM v_employees_info
          WHERE is_active ='false'
          ORDER BY start_date) a
          
          `}
      />
      <h4>Query 4 </h4>
      <p>
        1. Write a query that outputs only the top earner per position_title
        including first_name and position_title and their salary.
        <br />
      </p>
      <Code
        code={`
          SELECT
          first_name,
          position_title,
          salary
          FROM employees e1
          WHERE salary = (SELECT MAX(salary)
                  FROM employees e2
                  WHERE e1.position_title=e2.position_title)
          
          `}
      />
      <h4>Query 5 </h4>
      <p>
        Write a query that returns all meaningful aggregations of: sum of
        salary, number of employees, average salary grouped by all meaningful
        combinations of division, department, position_title.
      </p>
      <Code
        code={`
          SELECT 
          division,
          department,
          position_title,
          SUM(salary),
          COUNT(*),
          ROUND(AVG(salary),2)
          FROM employees
          NATURAL JOIN departments
          GROUP BY 
          ROLLUP(
          division,
          department,
          position_title
          )
          ORDER BY 1,2,3
          `}
      />
      <h4>Query 6 </h4>
      <p>
        Write a query that returns all employees (emp_id) including their
        position_title, department, their salary, and the rank of that salary
        partitioned by department. The highest salary per division should have
        rank 1.
      </p>
      <Code
        code={`
          SELECT
          emp_id,
          position_title,
          department,
          salary,
          RANK() OVER(PARTITION BY department ORDER BY salary DESC)
          FROM employees
          NATURAL LEFT JOIN department
          `}
      />
    </div>
  );
};

export default Content011;
