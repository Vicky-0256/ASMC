import React from "react";
import { createRoot } from "react-dom/client";
import ASMCPaperPage from "../asmc_web_animation.jsx";
import "./styles.css";

createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <ASMCPaperPage />
  </React.StrictMode>,
);
