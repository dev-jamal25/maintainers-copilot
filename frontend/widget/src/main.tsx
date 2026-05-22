import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { Widget } from "./Widget";
import "./styles.css";

const container = document.getElementById("root");
if (container) {
  createRoot(container).render(
    <StrictMode>
      <Widget />
    </StrictMode>,
  );
}
