import React from "react";
import "./Navigation.css";
import logo from "../../assets/logo.svg";
import arrow from "../../assets/arrow.svg";

const Navigation = () => {
  return (
    <nav className="navigation">
      <ul className="navigation__link-section">
        <a href="/" className="text-reg navigation__link">
          Inicio
        </a>
        <a href="/Formulario" className="text-reg navigation__link">
          Diagnostico
        </a>
      </ul>
    </nav>
  );
};

export default Navigation;
