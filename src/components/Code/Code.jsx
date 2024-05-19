// import React, { useEffect } from "react";
// import Prism from "prismjs";

// export default function Code({ code, language }) {
//   useEffect(() => {
//     Prism.highlightAll();
//   }, []);
//   return (
//     <div className="Code container">
//       <pre>
//         <code className={`language-${language}`}>{code}</code>
//       </pre>
//     </div>
//   );
// }

import hljs from "highlight.js/lib/core";
import python from "highlight.js/lib/languages/python";
import "highlight.js/styles/night-owl.css";
import "./code.css"

import { useEffect, useRef } from "react";

hljs.registerLanguage("python", python);

const Code = ({code}) => {
  const codeRef = useRef(null);

  useEffect(() => {
    hljs.highlightBlock(codeRef.current);
  }, []);

  return (
    <pre className="">
      <code
        className="python rounded  "
        style = {{color:'white'}}
        ref={codeRef}
      >
        {code}
      </code>
    </pre>
  );
};

export default Code;
