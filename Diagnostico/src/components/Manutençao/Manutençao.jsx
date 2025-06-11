import react from 'react';
import './Manutençao.css';

const Manutençao = () => {
    return(
        <div className='table'>
            <h2>Relação entre Tipo de Falha e <br /> Gases Dissolvidos no Óleo</h2>
            <table className='tabela'>
                <tr>
                    <td>Tipo de Falha</td>
                    <td>H₂</td>
                    <td>CH₄</td>
                    <td>C₂H₆</td>
                    <td>C₂H₄</td>
                    <td>C₂H₂</td>
                    <td>CO</td>
                    <td>CO₂</td>
                    <td>Observação</td>
                </tr>
                <tr>
                    <td>Descarga Parcial</td>
                    <td>Maior que 100</td>
                    <td>Menor que 50</td>
                    <td>Menor que 40</td>
                    <td>Menor que 20</td>
                    <td>0</td>
                    <td>Menor que 50</td>
                    <td>Menor que 100</td>
                    <td>H₂ elevado, demais baixos, ausência de acetileno</td>
                </tr>
                <tr>
                    <td>Descarga Parcial</td>
                    <td>Maior que 100</td>
                    <td>Menor que 50</td>
                    <td>Menor que 40</td>
                    <td>Menor que 20</td>
                    <td>0</td>
                    <td>Menor que 50</td>
                    <td>Menor que 100</td>
                    <td>H₂ elevado, demais baixos, ausência de acetileno</td>
                </tr>
                <tr>
                    <td>Descarga Parcial</td>
                    <td>Maior que 100</td>
                    <td>Menor que 50</td>
                    <td>Menor que 40</td>
                    <td>Menor que 20</td>
                    <td>0</td>
                    <td>Menor que 50</td>
                    <td>Menor que 100</td>
                    <td>H₂ elevado, demais baixos, ausência de acetileno</td>
                </tr>
                <tr>
                    <td>Descarga Parcial</td>
                    <td>Maior que 100</td>
                    <td>Menor que 50</td>
                    <td>Menor que 40</td>
                    <td>Menor que 20</td>
                    <td>0</td>
                    <td>Menor que 50</td>
                    <td>Menor que 100</td>
                    <td>H₂ elevado, demais baixos, ausência de acetileno</td>
                </tr>
                <tr>
                    <td>Descarga Parcial</td>
                    <td>Maior que 100</td>
                    <td>Menor que 50</td>
                    <td>Menor que 40</td>
                    <td>Menor que 20</td>
                    <td>0</td>
                    <td>Menor que 50</td>
                    <td>Menor que 100</td>
                    <td>H₂ elevado, demais baixos, ausência de acetileno</td>
                </tr>
                <tr>
                    <td>Descarga Parcial</td>
                    <td>Maior que 100</td>
                    <td>Menor que 50</td>
                    <td>Menor que 40</td>
                    <td>Menor que 20</td>
                    <td>0</td>
                    <td>Menor que 50</td>
                    <td>Menor que 100</td>
                    <td>H₂ elevado, demais baixos, ausência de acetileno</td>
                </tr>
            </table>
        </div>
    );
};

export default Manutençao;