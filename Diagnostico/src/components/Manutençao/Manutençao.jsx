import React from 'react';
import './Manutençao.css';

const Manutencao = () => {
  return (
    <div className="manutencao-container">
      <h2 className="titulo">
        Relação entre Tipo de Falha e <br /> Gases Dissolvidos no Óleo
      </h2>
      <div className="tabela-wrapper">
        <table className="tabela">
          <thead>
            <tr>
              <th>Tipo de Falha</th>
              <th>H₂</th>
              <th>CH₄</th>
              <th>C₂H₆</th>
              <th>C₂H₄</th>
              <th>C₂H₂</th>
              <th>CO</th>
              <th>CO₂</th>
              <th>Observação</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>Descarga Parcial</td>
              <td>&gt; 100</td>
              <td>&lt; 50</td>
              <td>&lt; 40</td>
              <td>&lt; 20</td>
              <td>0</td>
              <td>&lt; 50</td>
              <td>&lt; 100</td>
              <td>H₂ elevado, demais baixos, ausência de acetileno</td>
            </tr>
            <tr>
              <td>Superaquecimento do Óleo</td>
              <td>&lt; 100</td>
              <td>50–100</td>
              <td>40–100</td>
              <td>&lt; 50</td>
              <td>0</td>
              <td>&lt; 50</td>
              <td>&lt; 150</td>
              <td>CH₄ e C₂H₆ elevados, sem acetileno</td>
            </tr>
            <tr>
              <td>Superaquecimento Severo</td>
              <td>&lt; 150</td>
              <td>&gt; 100</td>
              <td>&gt; 100</td>
              <td>&gt; 100</td>
              <td>0</td>
              <td>&lt; 70</td>
              <td>&lt; 200</td>
              <td>Etileno predomina com CH₄ e C₂H₆, sem acetileno</td>
            </tr>
            <tr>
              <td>Degradação do Papel Isolante</td>
              <td>&lt; 100</td>
              <td>&lt; 50</td>
              <td>&lt; 40</td>
              <td>&lt; 20</td>
              <td>0</td>
              <td>&gt; 200</td>
              <td>&lt; 350</td>
              <td>CO e CO₂ elevados, relacionados ao envelhecimento do papel</td>
            </tr>
            <tr>
              <td>Arco Elétrico</td>
              <td>&gt; 150</td>
              <td>&gt; 100</td>
              <td>&lt; 50</td>
              <td>&gt; 100</td>
              <td>&gt; 1</td>
              <td>&lt; 70</td>
              <td>&lt; 200</td>
              <td>Acetileno presente, com picos de H₂ e C₂H₄</td>
            </tr>
            <tr>
              <td>Descarga de Baixa Energia</td>
              <td>100–200</td>
              <td>&lt; 80</td>
              <td>&lt; 60</td>
              <td>&lt; 50</td>
              <td>0.1–1</td>
              <td>&lt; 50</td>
              <td>&lt; 150</td>
              <td>Acetileno presente, com picos de H₂ e C₂H₄</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default Manutencao;
