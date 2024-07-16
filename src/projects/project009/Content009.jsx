import Code from "../../components/Code/Code.jsx";
import mainImg from "./mainImg009.png";
import img01 from "./img01.png";

const Content009 = () => {
  return (
    <div className="page container mt-5 mx-auto w-75 px-5 ">
      <h1 className="text-center">PostgreSQL</h1>
      <div className="text-center">
        <img src={mainImg} className="h-50 w-50"></img>
      </div>
      <p>
        Quick read, showing how to install and setup pgAdim 4, create databases
        and tables for querying
      </p>
      <h4>Download and setup</h4>
      <p>
        Installing postgresql and pgadmin is very straigforward. Find the right
        version of Postgres on the Official Website for Postgres for your
        current windows version (64bit or 32bit platform). Once the setup file
        has been downloaded, double-click on the file. It will run the
        Installation Wizard for PostgreSQL. From there it is very common setup
        process and click next in setup wizard. Then, you can choose the
        component you want to install on your workstation. You can choose any of
        the following but we usually choose all of it when we are going to
        install PostgreSQL for the first time.
        <br />
        <br />
      </p>
      <div className="text-center">
        <img
          src={img01}
          alt="pgadmin"
          style={{ height: "300px", width: "300px" }}
        />
      </div>
      <p>
        PostgreSQL Server pgAdmin4: It is a graphical interface that is used to
        manage the PostgreSQL database Stack builder: The stack builder will be
        used to download and install drivers and additional tools. Command-line
        tools like pg_bench, pg_restore, pg_basebackup, libpq, pg_dump, and
        pg_restore. Then you can setup a password to secure loing to your
        servers. After go through next steps. PostgreSQL server will have been
        installed successfully. You can add or install additional components and
        drivers if you want to. Click on Finish to complete the installation.
        <br />
        To verify the installation, search for psql shell in your operating
        system and after opening it enter necessary details (or just press enter
        for all) and password to start the server, then you can type SELECT
        version() command to get vesion info.
        <br />
      </p>
      <h4>Database Creation</h4>
      <p>
        Database in pgAdmin There are two ways to create a database in the
        listening server the esiest way is to start pgAdmin, go to the server
        and click on database then creat database and give name that's it.
        <br />
        Another way of creating database is from sql shell. go to shell and
        connect serve by give required details and password, then type command
        CREATE DATABASE db_name ; and also if you want to sepicfy oher details
        yo can type command CREATE DATABASE db_name WITH ENCODING 'UTF8'
        LC_COLLATE='any_locale' LC_CTYPE='any_collation'; To connect to the
        database type command \c db_name and tolyst all command type \l
      </p>

      <h4>Data types </h4>
      <p>
        PostgreSQL offers a wide range of data types to efficiently store and
        manipulate various kinds of data. Understanding these data types is
        crucial for optimizing database performance and ensuring data integrity.
        Here, we will look into the various data types available in PostgreSQL.
        The following data types are supported by PostgreSQL:
        <br />
        <br />
        Boolean Type: [ hold true, false, and null values] <br />
        Character Type: [ Types such as char, varchar, and text]
        <br />
        Numeric Type: [ Types such as integer and floating-point number]
        <br />
        Temporal Type: [ Types such as date, time, timestamp, and interval]
        <br />
        UUID Type: [ for storing UUID (Universally Unique Identifiers) ]<br />
        Array Type: [ for storing array strings, numbers, etc.]
        <br />
        JSON Type: [ stores JSON data]
        <br />
        hstore Type: [ stores key-value pair]
        <br />
        Special Type: [ Types such as network address and geometric data]
        <br />
        
        <br />I have listed the most useful data types below: Numeric: int
        (4bytes), samllint (2 bytes), bigint(8 bytes), decimal[(p,s)] exact
        numeric of selectable precision, serial (1,2,3,...)
        <br />
        String: varchar (n) (variable-length character string) , char (n)
        (fixed-length character string), text (variable-length character
        string).
        <br /> Date/time: timestamp, timestamptz, date, time, timetz(with time
        zone.)
      </p>
      <h4> Constraints </h4>
      <p>
        Constraints are the rules enforced on data columns on table. These are
        used to prevent invalid data from being entered into the database. This
        ensures the accuracy and reliability of the data in the database.
        Constraints could be column level or table level. Column level
        constraints are applied only to one column whereas table level
        constraints are applied to the whole table. Defining a data type for a
        column is a constraint in itself. For example, a column of type DATE
        constrains the column to valid dates. The following are commonly used
        constraints available in PostgreSQL.
        <br />
        NOT NULL Constraint − Ensures that a column cannot have NULL value.
        <br />
        UNIQUE Constraint − Ensures that all values in a column are different.
        <br />
        PRIMARY Key − Uniquely identifies each row/record in a database table.
        <br />
        FOREIGN Key − Constrains data based on columns in other tables.
        <br />
        CHECK Constraint − The CHECK constraint ensures that all values in a
        column satisfy certain conditions.
        <br />
        EXCLUSION Constraint − The EXCLUDE constraint ensures that if any two
        rows are compared on the specified column(s) or expression(s) using the
        specified operator(s), not all of these comparisons will return TRUE.
        <br />
        
        
      </p>
      <h4> Tabel Creation </h4>
      <p>
        CREATE TABLE is a keyword, telling the database system to create a new
        table. The unique name or identifier for the table follows the CREATE
        TABLE statement. Initially, the empty table in the current database is
        owned by the user issuing the command. Basic syntax of CREATE TABLE
        statement is as follows − <br />
        CREAT TABLE tb_name(
        <br />
        col01 d_type constraint
        <br />
        col02 d-type constraint
        <br />
        .<br />
        PRAIMARY KEY (col01 or more)
        <br />
        )
        <br />
        
      </p>
      <h4> Data insertion </h4>
      <p>
        SQL query "INSERT INTO TABLE_NAME (column1, column2, column3,...columnN)
        VALUES (value1, value2, value3,...valueN);" is used to insert values
        into tabel row by row. If the data is in csv files you can easily import
        data into table using pgAdimn user interface.
        <br />
        <br />
      </p>
    </div>
  );
};

export default Content009;
