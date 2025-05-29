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
          <p className="text-reg features__subheading">
            A técnica mais difundida para o diagnóstico dessas falhas é a Análise de Gases Dissolvidos (DGA), 
            que interpreta os gases produzidos pela decomposição do óleo isolanteAtravés dos gases dissolvidos 
            no óleo isolante do transformador, é possível avaliar o estado do seu isolamento com a indicação de 
            possíveis estados de falhas que possam estar ocorrendo.
          </p>
          <p className="text-reg features__subheading">
            Esses processos de análise de gases dissolvidos 
            ou análise cromatográfica do óleo isolante são realizados a partir da amostra do óleo isolante de um 
            transformador em operação, quantificando a concentração de certos gases gerados no equipamento e 
            dissolvidos no óleo isolante, indicando a presença de falhas incipientes associadas à parte ativa do 
            transformador.
          </p>
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

        <div className="divCollumn">
          <h2 className="h2 features__heading">
            <br/>
            <br/>
            Análise dos tipos de {" "}
            <span className="h2 features__text-gradient">Falhas</span>
          </h2>
        </div>  
            
        <div className="features__feature-container2">
          <div className="feature">
            <h2 className="h3 features__heading">Descarga Parcial</h2>
            <p className="text-reg features__subheading">
              A descarga parcial é uma falha de baixa energia que ocorre em áreas isolantes com imperfeições, gerando principalmente hidrogênio. É considerada de risco baixo a moderado, pois indica o início de um processo de degradação.
            </p>
          </div>
          <div className="feature">
            <h2 className="h3 features__heading">Superaquecimento do óleo</h2>
            <p className="text-reg features__subheading">
              O superaquecimento do óleo, quando leve ou moderado, ocorre em temperaturas entre 150 e 300 °C e gera metano e etano. Essa condição sugere pontos quentes no sistema e representa um risco moderado.
            </p>
          </div>
        </div>

        <div className="features__feature-container2">
          <div className="feature">
            <h2 className="h3 features__heading">Superaquecimento do papel isolante</h2>
            <p className="text-reg features__subheading">
              A degradação do papel isolante, por sua vez, ocorre a partir de 150 °C e libera grandes quantidades de monóxido e dióxido de carbono. Esse tipo de falha está relacionado ao envelhecimento do isolamento sólido, representando um risco elevado e muitas vezes irreversível. 
            </p>
          </div>
          <div className="feature">
            <h2 className="h3 features__heading">Superaquecimento severo do óleo</h2>
            <p className="text-reg features__subheading">
              O superaquecimento severo do óleo ocorre quando a temperatura no interior do transformador ultrapassa os 300 °C, geralmente devido a falhas em conexões, sobrecarga ou problemas de ventilação. Nessa condição, o óleo isolante se decompõe termicamente, liberando grandes quantidades de etileno, além de metano e etano em menores proporções.
            </p>
          </div>
        </div>

        <div className="features__feature-container2">
          <div className="feature">
            <h2 className="h3 features__heading">Arco Elétrico</h2>
            <p className="text-reg features__subheading">
              O arco elétrico é uma falha muito mais grave, causada por descargas elétricas intensas no interior do equipamento, como curtos-circuitos ou contatos soltos. Ele ocorre em temperaturas superiores a 700 °C e leva à decomposição explosiva do óleo e, muitas vezes, também do papel isolante.
            </p>
          </div>
          <div className="feature">
            <h2 className="h3 features__heading">Descarga de baixa energia</h2>
            <p className="text-reg features__subheading">
              A descarga de baixa energia, como faíscas ou efeito corona, ocorre entre 400 e 700 °C e também gera hidrogênio, além de pequenas quantidades de acetileno. Embora não tão crítica quanto o arco, essa condição exige atenção, pois pode evoluir rapidamente.
            </p>
          </div>
        </div>

    </section>
  );
};

export default Features;
