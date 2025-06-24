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
              <th>C₂H₂</th>
              <th>C₂H₄</th>
              <th>C₂H₆</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>Sem fallha</td>
              <td>13 - 19</td>
              <td>1 - 3</td>
              <td>1 - 1</td>
              <td>1 - 21</td>
              <td>0 - 6</td>
            </tr>
            <tr>
              <td>Descarga Parcial</td>
              <td>0 - 2587</td>
              <td>0 - 100</td>
              <td>0 - 7.1</td>
              <td>0 - 43</td>
              <td>0 - 127</td>
            </tr>
            <tr>
              <td>Superaquecimento Leve</td>
              <td>4 - 92600</td>
              <td>1 - 10200</td>
              <td>0 - 96</td>
              <td>0 - 66</td>
              <td>0 - 745</td>
            </tr>
            <tr>
              <td>Superaquecimento Grave</td>
              <td>35 - 40280</td>
              <td>6 - 18342</td>
              <td>0 - 2111</td>
              <td>0 - 304</td>
              <td>0 - 670</td>
            </tr>
          </tbody>
        </table>
      </div>
      <h2 className="titulo">
        Prevenção e Manutenção por Tipo de Falha
      </h2>
      <div className="tabela-wrapper">
        <table className="tabela">
          <thead>
            <tr>
              <th>Tipo de Falha</th>
              <th>Prevenção</th>
              <th>Manutenção</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>Descarga parcial</td>
              <td>Manter o isolamento seco e íntegro; controlar a umidade no óleo; usar materiais isolantes de qualidade; monitoramento contínuo com sensores de DP em equipamentos críticos.</td>
              <td>Análise frequente de DGA (principalmente H₂); testes de rigidez dielétrica; inspeção interna se necessário; secagem do óleo e do isolamento; reaplicação de óleo com desgasificação.</td>
            </tr>
            <tr>
              <td>Superaquecimento Leve</td>
              <td>Evitar sobrecargas; garantir boas conexões elétricas; manter o sistema de resfriamento em boas condições; executar estudos de carregamento.</td>
              <td>Termografia regular; reaperto e limpeza de conexões; substituição de terminais desgastados; verificação da ventilação e da circulação de óleo.</td>             
            </tr>
            <tr>
              <td>Superaquecimento Grave</td>
              <td>Monitoramento contínuo de temperatura; manutenção do sistema de resfriamento; evitar sobrecargas prolongadas; instalar alarmes térmicos.</td>
              <td>Purificação ou troca do óleo; testes físico-químicos do óleo; inspeção de enrolamentos e partes internas; reparo ou substituição de componentes danificados.</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default Manutencao;
