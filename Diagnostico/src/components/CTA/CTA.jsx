import React from "react";
import "./CTA.css";
import ctaShapes from "../../assets/cta-shapes.png";
import Descarga from "../../assets/Descarga.png";
import Oleo from "../../assets/Oleo.png";
import Papel from "../../assets/Papel.png";
import Superaquecimento from "../../assets/Superaquecimento.png";
import Arco from "../../assets/Arco.png";
import Energia from "../../assets/Energia.png";
import arrow from "../../assets/arrow.svg";
import circleGradient from "../../assets/circle-gradient.png";

const CTA = () => {
  return (
    <section className="cta">
      <div className="cta-content">
        <div className="cta-content__left">
          <h3 className="h3 cta__heading">
            Descarga Parcial
          </h3>
          <p className="text-reg cta__description">
            A descarga parcial é uma falha de baixa energia que ocorre em áreas isolantes com imperfeições, gerando principalmente hidrogênio. É considerada de risco baixo a moderado, pois indica o início de um processo de degradação.
          </p>
        </div>
        <div className="cta-content__right">
          <img className="cta__shapes" src={Descarga} alt="graphic shapes" />
        </div>
      </div>

      <div className="cta-content">
        <div className="cta-content__right">
          <img className="cta__shapes" src={Oleo} alt="graphic shapes" />
        </div>
        <div className="cta-content__left">
          <h3 className="h3 cta__heading">
            Superaquecimento do óleo
          </h3>
          <p className="text-reg cta__description">
            O superaquecimento do óleo, quando leve ou moderado, ocorre em temperaturas entre 150 e 300 °C e gera metano e etano. Essa condição sugere pontos quentes no sistema e representa um risco moderado.
          </p>
        </div>
      </div>
      <img
        className="cta__circle-gradient"
        src={circleGradient}
        alt="circle gradient"
      />

      <div className="cta-content">
        <div className="cta-content__left">
          <h3 className="h3 cta__heading">
            Superaquecimento do papel isolante
          </h3>
          <p className="text-reg cta__description">
            A degradação do papel isolante, por sua vez, ocorre a partir de 150 °C e libera grandes quantidades de monóxido e dióxido de carbono. Esse tipo de falha está relacionado ao envelhecimento do isolamento sólido, representando um risco elevado e muitas vezes irreversível. 
          </p>
        </div>
        <div className="cta-content__right">
          <img className="cta__shapes" src={Papel} alt="graphic shapes" />
        </div>
      </div>
      <img
        className="cta__circle-gradient"
        src={circleGradient}
        alt="circle gradient"
      />

      <div className="cta-content">
        <div className="cta-content__right">
          <img className="cta__shapes" src={Superaquecimento} alt="graphic shapes" />
        </div>
        <div className="cta-content__left">
          <h3 className="h3 cta__heading">
            Superaquecimento severo do óleo
          </h3>
          <p className="text-reg cta__description">
            O superaquecimento severo do óleo ocorre quando a temperatura no interior do transformador ultrapassa os 300 °C, geralmente devido a falhas em conexões, sobrecarga ou problemas de ventilação. Nessa condição, o óleo isolante se decompõe termicamente, liberando grandes quantidades de etileno, além de metano e etano em menores proporções.
          </p>
        </div>
      </div>
      <img
        className="cta__circle-gradient"
        src={circleGradient}
        alt="circle gradient"
      />

      <div className="cta-content">
        <div className="cta-content__left">
          <h3 className="h3 cta__heading">
            Arco Elétrico
          </h3>
          <p className="text-reg cta__description">
            O arco elétrico é uma falha muito mais grave, causada por descargas elétricas intensas no interior do equipamento, como curtos-circuitos ou contatos soltos. Ele ocorre em temperaturas superiores a 700 °C e leva à decomposição explosiva do óleo e, muitas vezes, também do papel isolante.
          </p>
        </div>
        <div className="cta-content__right">
          <img className="cta__shapes" src={Arco} alt="graphic shapes" />
        </div>
      </div>
      <img
        className="cta__circle-gradient"
        src={circleGradient}
        alt="circle gradient"
      />

      <div className="cta-content">
        <div className="cta-content__right">
          <img className="cta__shapes" src={Energia} alt="graphic shapes" />
        </div>
        <div className="cta-content__left">
          <h3 className="h3 cta__heading">
            Descarga de baixa energia
          </h3>
          <p className="text-reg cta__description">
            A descarga de baixa energia, como faíscas ou efeito corona, ocorre entre 400 e 700 °C e também gera hidrogênio, além de pequenas quantidades de acetileno. Embora não tão crítica quanto o arco, essa condição exige atenção, pois pode evoluir rapidamente.
          </p>
        </div>
      </div>
      <img
        className="cta__circle-gradient"
        src={circleGradient}
        alt="circle gradient"
      />
    </section>
  );
};

export default CTA;
