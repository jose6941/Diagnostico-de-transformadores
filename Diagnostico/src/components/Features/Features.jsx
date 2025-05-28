import React from "react";
import "./Features.css";
import { features } from "../../utils/constants";

const Features = () => {
  return (
    <section className="features">
      <div className="features__heading-container">
        <h2 className="h2 features__heading">
          Sobre o {" "}
          <span className="h2 features__text-gradient">Projeto</span>
        </h2>
        <p className="text-reg features__subheading">
          <br/>
          A técnica mais difundida para o diagnóstico dessas falhas é a Análise de Gases Dissolvidos (DGA), 
          que interpreta os gases produzidos pela decomposição do óleo isolanteAtravés dos gases dissolvidos 
          no óleo isolante do transformador, é possível avaliar o estado do seu isolamento com a indicação de 
          possíveis estados de falhas que possam estar ocorrendo.
          <br/><br/> 
          Esses processos de análise de gases dissolvidos 
          ou análise cromatográfica do óleo isolante são realizados a partir da amostra do óleo isolante de um 
          transformador em operação, quantificando a concentração de certos gases gerados no equipamento e 
          dissolvidos no óleo isolante, indicando a presença de falhas incipientes associadas à parte ativa do 
          transformador.
        </p>
      </div>
      <div className="features__feature-container">
         <h2 className="h2 features__heading">
          Análise de {" "}
          <span className="h2 features__text-gradient">Gases</span>
        </h2>
         <p className="text-reg features__subheading">
          <br/>
            A técnica mais difundida para o diagnóstico dessas falhas é a Análise de Gases Dissolvidos (DGA), 
            que interpreta os gases produzidos pela decomposição do óleo isolanteAtravés dos gases dissolvidos 
            no óleo isolante do transformador, é possível avaliar o estado do seu isolamento com a indicação de 
            possíveis estados de falhas que possam estar ocorrendo.
          </p>
        {features.map((obj, i) => {
          return (
            <div className={`feature ${obj.gridArea}`} key={i}>
              <img
                className="feature__icon"
                src={obj.image}
                alt={obj.heading}
              />
              <p className="text-large feature__heading">{obj.heading}</p>
              <p className="text-reg feature__description">{obj.description}</p>
            </div>
          );
        })}
      </div>
      <div className="features__overlay-gradient"></div>
    </section>
  );
};

export default Features;
