import React from 'react';
import { useLocation, useNavigate } from 'react-router-dom';

const Resultados = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const resultado = location.state;

  if (!resultado) {
    return (
      <div style={{ padding: '2rem', color: 'white', backgroundColor: '#111', minHeight: '100vh' }}>
        <h2>Nenhum resultado disponível</h2>
        <button onClick={() => navigate('/formulario')} style={{ marginTop: '20px' }}>
          Voltar para o formulário
        </button>
      </div>
    );
  }

  return (
    <div style={styles.container}>
      <h2>Resultado do Diagnóstico</h2>
      <div style={styles.divResultado}>
        <div style={styles.falha}>
          <h2 style={styles.h2}>Falha Identificada</h2>
          <h1 style={styles.h1}>{resultado.falha}</h1>
        </div>
        
        <p>Descrição: {resultado.descricao}</p>
        <p>Medidas: {resultado.medidas}</p>
      </div>
    </div>
  );
};

const styles = {
    container: {
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'center',
        alignItems: 'center',
        columnGap: '101px',
        marginTop: '20px',
        marginBottom: '20px',
        color: 'white'
    },
    divResultado: {
      display: 'flex',
      flexDirection: 'column',  
      alignItems: 'center',
      width: '100%',
      maxWidth: '750px',
      backgroundColor: '#111',
      borderRadius: '6px',
      fontFamily: 'Courier New, monospace',
    },
    h1: {
        margin: 0
    },
    h2: {
        marginTop: '30px',
        marginBottom: '20px'
    },
    falha: {
        marginBottom: '40px',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center'
    }
}

export default Resultados;
