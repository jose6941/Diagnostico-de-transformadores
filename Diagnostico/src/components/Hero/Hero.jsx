import React from "react";
import "./Hero.css";
import arrow from "../../assets/arrow.svg";
import abstractShapes from "../../assets/abstract-shapes.png";

const Hero = () => {
  return (
    <section className="hero">
      <div className="hero__column">
        <h1 className="h1 hero__heading">
          <span className="hero__heading-gradient">Diagnostico</span>
          de {" "}
          <span className="hero__heading-gradient"> Transformadores</span>
           de potência
        </h1>
        <p className="text-reg hero__subheading">
          Diagnóstico inteligente de falhas em transformadores de potência com base em DGA, Análise de Gases Dissolvidos.
        </p>
      </div>
      <div className="hero__column">
        <img
          className="hero__graphic"
          src={abstractShapes}
          alt="abstract shapes"
        />
      </div>
    </section>
  );
};

export default Hero;
