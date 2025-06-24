import React, { useState } from 'react';
import { ArrowLeft } from 'lucide-react';
import { useLocation, useNavigate } from 'react-router-dom';

const Resultados = () => {
  const location = useLocation();
  const resultado = location.state;

  const navigate = (path) => {
    console.log(`Navegando para: ${path}`);
  };

  if (!resultado) {
    return (
      <div style={styles.noResultContainer}>
        <div style={styles.noResultCard}>
          <h2 style={styles.noResultTitle}>Nenhum resultado disponível</h2>
          <button 
            onClick={() => navigate('/formulario')} 
            style={styles.backButton}
            onMouseEnter={(e) => e.target.style.backgroundColor = '#1d4ed8'}
            onMouseLeave={(e) => e.target.style.backgroundColor = '#2563eb'}
          >
            <ArrowLeft style={styles.buttonIcon} />
            Voltar para o formulário
          </button>
        </div>
      </div>
    );
  }

  return (
    <div style={styles.container}>
      <div style={styles.wrapper}>
        <div style={styles.header}>
          <h2 style={styles.mainTitle}>Resultado do Diagnóstico</h2>
        </div>
        
        <div style={styles.resultCard}>
          <div style={styles.failureSection}>
            <h2 style={styles.sectionTitle}>Falha Identificada</h2>
            <h1 style={styles.failureTitle}>{resultado.falha}</h1>
          </div>
          
          <div style={styles.contentSection}>
            <div style={styles.descriptionBox}>
              <h3 style={styles.contentTitle}>Descrição:</h3>
              <p style={styles.contentText}>{resultado.descricao}</p>
            </div>
            
            <div style={styles.measuresBox}>
              <h3 style={styles.contentTitle}>Medidas:</h3>
              <p style={styles.contentText}>{resultado.medidas}</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

const styles = {
  // Container principal
  container: {
    minHeight: '100vh',
    padding: '2rem 1rem',
    fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif'
  },
  
  wrapper: {
    maxWidth: '64rem',
    margin: '0 auto'
  },
  
  // Header
  header: {
    textAlign: 'center',
    marginBottom: '4rem',
    marginTop: '2rem'
  },
  
  mainTitle: {
    fontSize: '2.5rem',
    fontWeight: 'bold',
    color: '#ffffff',
    margin: 0
  },
  
  // Card principal
  resultCard: {
    backgroundColor: '#111',
    borderRadius: '0.5rem',
    border: '1px solid #475569',
    padding: '2rem',
    maxWidth: '48rem',
    margin: '0 auto'
  },
  
  // Seção da falha
  failureSection: {
    textAlign: 'center',
    marginBottom: '2rem'
  },
  
  sectionTitle: {
    fontSize: '1.25rem',
    fontWeight: '600',
    color: '#cbd5e1',
    marginBottom: '1.5rem',
    margin: '0 0 1.5rem 0'
  },
  
  failureTitle: {
    fontSize: '2.5rem',
    fontWeight: 'bold',
    color: '#ffffff',
    marginBottom: '3rem',
    margin: '0 0 1.5rem 0'
  },
  
  // Seção de conteúdo
  contentSection: {
    display: 'flex',
    flexDirection: 'column',
    gap: '1.5rem'
  },
  
  descriptionBox: {
    borderLeft: '4px solid #3b82f6',
    paddingLeft: '1.5rem'
  },
  
  measuresBox: {
    borderLeft: '4px solid #10b981',
    paddingLeft: '1.5rem'
  },
  
  contentTitle: {
    fontSize: '1.5rem',
    fontWeight: '600',
    color: '#cbd5e1',
    marginBottom: '0.75rem',
    margin: '0 0 0.75rem 0'
  },
  
  contentText: {
    color: '#e2e8f0',
    lineHeight: '1.625',
    margin: 0
  },
  
  // Estilos para caso sem resultado
  noResultContainer: {
    minHeight: '100vh',
    backgroundColor: '#0f172a',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    padding: '1.5rem'
  },
  
  noResultCard: {
    backgroundColor: '#1e293b',
    borderRadius: '0.5rem',
    border: '1px solid #475569',
    padding: '2rem',
    maxWidth: '28rem',
    width: '100%',
    textAlign: 'center'
  },
  
  noResultTitle: {
    fontSize: '2rem',
    fontWeight: '600',
    color: '#ffffff',
    marginBottom: '1rem',
    margin: '0 0 1rem 0'
  },
  
  backButton: {
    backgroundColor: '#2563eb',
    color: '#ffffff',
    fontWeight: '500',
    padding: '0.75rem 1.5rem',
    borderRadius: '0.5rem',
    border: 'none',
    cursor: 'pointer',
    transition: 'background-color 0.2s',
    display: 'flex',
    alignItems: 'center',
    gap: '0.5rem',
    margin: '1.25rem auto 0',
    fontSize: '1rem'
  },
  
  buttonIcon: {
    width: '1rem',
    height: '1rem'
  }
};

export default Resultados;