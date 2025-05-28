import React from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import FormularioDiagnostico from './FormularioDiagnostico';
import { motion } from 'framer-motion';
import Chart from 'react-apexcharts';

const Home = () => (
  <motion.section
    style={{ ...landingStyles.section}}
    initial={{ opacity: 0 }}
    animate={{ opacity: 1 }}
    transition={{ duration: 0.6 }}
  >
    <div style={landingStyles.overlay}>
      <h1 style={landingStyles.heading}>Análise de Transformadores por IA</h1>
      <p style={landingStyles.text}>
        Diagnóstico inteligente de falhas em transformadores de potência com base em DGA (Análise de Gases Dissolvidos).
      </p>
      <Link to="/formulario" style={landingStyles.button}>Fazer Diagnóstico</Link>
    </div>
  </motion.section>
);

const Sobre = () => (
  <motion.section
    style={landingStyles.sobreWrapper}
    initial={{ opacity: 0 }}
    animate={{ opacity: 1 }}
    transition={{ duration: 0.6 }}
  >
    <div style={landingStyles.sobreContent}>
      <h2 style={landingStyles.pageTitle}>Informações do Projeto</h2>

      <div style={landingStyles.cardSection}>
        <div style={landingStyles.cardCentered}>
          <h3 style={landingStyles.headingLight}>Sobre o Projeto</h3>
          <p style={landingStyles.textLight}>
            Este projeto utiliza inteligência artificial para diagnosticar falhas incipientes em transformadores de potência por meio da análise de gases dissolvidos (DGA).
          </p>
        </div>
        <div style={landingStyles.cardCentered}>
          <h3 style={landingStyles.headingLight}>Análise DGA</h3>
          <p style={landingStyles.textLight}>
            A Análise de Gases Dissolvidos mede os gases no óleo isolante, originados por falhas térmicas ou elétricas internas nos transformadores.
          </p>
        </div>
      </div>

      <h3 style={landingStyles.sectionTitle}>Gases e Suas Influências</h3>
      <div style={landingStyles.cardSection}>
        <div style={landingStyles.cardCentered}>
          <ul style={landingStyles.list}>
            <li><strong>Hidrogênio (H₂):</strong> Indica descargas parciais.</li>
            <li><strong>Metano (CH₄):</strong> Associado a aquecimento do óleo.</li>
            <li><strong>Etano (C₂H₆) e Etileno (C₂H₄):</strong> Relacionados ao superaquecimento.</li>
            <li><strong>Acetileno (C₂H₂):</strong> Indica arco elétrico.</li>
          </ul>
        </div>
        <div style={landingStyles.cardCentered}>
          <Chart
            options={{ labels: ['H₂', 'CH₄', 'C₂H₂', 'C₂H₄', 'C₂H₆'], legend: { position: 'bottom' }, colors: ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'] }}
            series={[30, 15, 10, 25, 20]}
            type="pie"
            width={300}
          />
        </div>
      </div>

      <h3 style={landingStyles.sectionTitle}>Tipos de Falhas Detectáveis</h3>
      <div style={landingStyles.cardSection}>
        <div style={landingStyles.cardCentered}>
          <ul style={landingStyles.list}>
            <li><strong>Descarga Parcial:</strong> Pequenas faíscas internas, sinal de futuras falhas.</li>
            <li><strong>Arco Elétrico:</strong> Falha crítica por descarga intensa.</li>
            <li><strong>Aquecimento:</strong> De baixa ou alta temperatura devido à sobrecarga.</li>
            <li><strong>Curto entre Espiras:</strong> Falhas internas entre enrolamentos.</li>
          </ul>
        </div>
        <div style={landingStyles.cardCentered}>
          <p style={landingStyles.textLight}>
            A detecção precoce evita danos graves e prolonga a vida útil do transformador, reduzindo custos e interrupções.
          </p>
        </div>
      </div>
    </div>
  </motion.section>
);

const Navbar = () => (
  <nav style={landingStyles.navbar}>
    <h2 style={landingStyles.logo}>DGA AI</h2>
    <div style={landingStyles.links}>
      <Link to="/" style={landingStyles.link}>Início</Link>
      <Link to="/sobre" style={landingStyles.link}>Sobre</Link>
      <Link to="/formulario" style={landingStyles.link}>Diagnóstico</Link>
    </div>
  </nav>
);

const App = () => (
  <Router>
    <div style={landingStyles.app}>
      <Navbar />
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/sobre" element={<Sobre />} />
        <Route path="/formulario" element={<FormularioDiagnostico />} />
      </Routes>
    </div>
  </Router>
);

const landingStyles = {
  app: {
    width: '100vw',
    minHeight: '100vh',
    display: 'flex',
    flexDirection: 'column',
    backgroundColor: '#0D1117',
    color: '#C9D1D9',
    fontFamily: 'Segoe UI, sans-serif',
  },
  navbar: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: '20px 40px',
    backgroundColor: '#090B10',
    borderBottom: '1px solid #30363D',
    position: 'sticky',
    top: 0,
    zIndex: 10,
  },
  logo: {
    color: '#58A6FF',
    fontSize: '24px',
    fontWeight: 'bold',
  },
  links: {
    display: 'flex',
    gap: '20px',
  },
  link: {
    color: '#C9D1D9',
    textDecoration: 'none',
    fontSize: '16px',
    transition: 'color 0.3s',
  },
  section: {
    backgroundSize: 'cover',
    backgroundPosition: 'center',
    height: 'calc(100vh - 80px)',
    display: 'flex',
    justifyContent: 'center',
    alignItems: 'center',
    color: '#ffffff',
    position: 'relative'
  },
  overlay: {
    backgroundColor: 'rgba(0, 0, 0, 0.6)',
    padding: '40px',
    borderRadius: '12px',
    textAlign: 'center',
    maxWidth: '600px'
  },
  heading: {
    fontSize: '40px',
    fontWeight: 'bold',
    marginBottom: '20px',
    color: '#58A6FF',
  },
  pageTitle: {
    fontSize: '32px',
    fontWeight: 'bold',
    marginBottom: '40px',
    textAlign: 'center',
    color: '#58A6FF',
  },
  sectionTitle: {
    fontSize: '26px',
    marginBottom: '20px',
    textAlign: 'center',
    color: '#58A6FF'
  },
  text: {
    fontSize: '18px',
    marginBottom: '30px',
    lineHeight: '1.6',
  },
  button: {
    backgroundColor: '#238636',
    padding: '12px 24px',
    borderRadius: '8px',
    textDecoration: 'none',
    fontWeight: 'bold',
    color: '#ffffff',
    fontSize: '16px',
    transition: 'background-color 0.3s',
  },
  sobreWrapper: {
    padding: '60px 40px',
    backgroundColor: '#161B22',
    color: '#C9D1D9'
  },
  sobreContent: {
    maxWidth: '1000px',
    margin: '0 auto',
    textAlign: 'center'
  },
  cardSection: {
    display: 'flex',
    gap: '20px',
    marginBottom: '40px',
    justifyContent: 'center',
    flexWrap: 'wrap'
  },
  card: {
    backgroundColor: '#0D1117',
    padding: '20px',
    borderRadius: '12px',
    flex: '1 1 calc(50% - 20px)',
    boxShadow: '0 0 10px rgba(0,0,0,0.3)'
  },
  cardCentered: {
    backgroundColor: '#0D1117',
    padding: '20px',
    borderRadius: '12px',
    width: '100%',
    maxWidth: '460px',
    boxShadow: '0 0 10px rgba(0,0,0,0.3)',
    margin: '0 auto'
  },
  headingLight: {
    fontSize: '20px',
    color: '#58A6FF',
    marginBottom: '10px',
  },
  textLight: {
    fontSize: '16px',
    lineHeight: '1.6',
    color: '#C9D1D9',
  },
  list: {
    marginTop: '10px',
    paddingLeft: '20px',
    lineHeight: '1.8',
    textAlign: 'left'
  }
};

export default App;
