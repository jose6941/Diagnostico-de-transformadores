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
        <div className="divRow">
          <div>
            <h2 className="h2 features__heading">Análise</h2>
            <p className="text-reg features__subheading">
              A técnica mais difundida para o diagnóstico dessas falhas é a Análise de Gases Dissolvidos (DGA), 
              que interpreta os gases produzidos pela decomposição do óleo isolanteAtravés dos gases dissolvidos 
              no óleo isolante do transformador, é possível avaliar o estado do seu isolamento com a indicação de 
              possíveis estados de falhas que possam estar ocorrendo.
            </p>
          </div>
          <div>
            <h2 className="h2 features__heading">Processo</h2>
            <p className="text-reg features__subheading">
              Esses processos de análise de gases dissolvidos 
              ou análise cromatográfica do óleo isolante são realizados a partir da amostra do óleo isolante de um 
              transformador em operação, quantificando a concentração de certos gases gerados no equipamento e 
              dissolvidos no óleo isolante, indicando a presença de falhas incipientes associadas à parte ativa do 
              transformador.
            </p>
          </div>
        </div>
        
        
        </div>
        <div className="divCollumn">
          <h2 className="h2 features__heading">
            Análise de {" "}
            <span className="h2 features__text-gradient">Gases</span>
          </h2>
        </div>  
            
        <div className="features__feature-container2">
          <div className="feature">
            <h2 className="h2 features__heading">H₂</h2>
            <p className="text-reg features__subheading">
              O hidrogênio (H₂) é o gás mais sensível e aparece em praticamente todos os tipos de falhas, especialmente em descargas parciais e pequenos arcos. Concentrações elevadas isoladas indicam que algo anormal está ocorrendo, mas ele não define o tipo de falha sozinho.
            </p>
          </div>
          <div className="feature">
            <h2 className="h2 features__heading">CH₄</h2>
            <p className="text-reg features__subheading">
              O metano (CH₄) e o etano (C₂H₆) são produzidos em condições de superaquecimento leve a moderado do óleo isolante (150–300 °C). Quando aparecem juntos, especialmente com predomínio de etano, indicam um aquecimento não muito severo.
            </p>
          </div>
        </div>

        <div className="features__feature-container3">
          <div className="feature">
            <h2 className="h2 features__heading">C₂H₂</h2>
            <p className="text-reg features__subheading">
              O acetileno (C₂H₂) é o gás mais crítico, pois só aparece em casos de arco elétrico de alta energia, com temperaturas superiores a 700 °C. Mesmo em pequenas quantidades, exige atenção imediata.
            </p>
          </div>
          <div className="feature">
            <h2 className="h2 features__heading">C₂H₄</h2>
            <p className="text-reg features__subheading">
              O etileno (C₂H₄) é gerado em temperaturas mais altas (acima de 300 °C) e está ligado a superaquecimento severo do óleo. Quando aparece em maior proporção que metano e etano, é sinal de uma falha térmica 
            </p>
          </div>
          <div className="feature">
            <h2 className="h2 features__heading">C₂H₆</h2>
            <p className="text-reg features__subheading">
              O etano (C₂H₆) também está relacionado à degradação térmica do óleo, comumente em temperaturas moderadas, entre 150 °C e 300 °C. É frequentemente observado juntamente com metano e em menor grau com etileno.
            </p>
          </div>
        </div>
    </section>
  );
};

export default Features;
