import React from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import FormularioDiagnostico from './FormularioDiagnostico';
import { motion } from 'framer-motion';
import bgImage from './assets/transformer.jpg';

const Home = () => (
  <motion.section
    style={{ ...landingStyles.section, backgroundImage: `url(${bgImage})` }}
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
    style={landingStyles.sectionLight}
    initial={{ opacity: 0 }}
    animate={{ opacity: 1 }}
    transition={{ duration: 0.6 }}
  >
    <h2 style={landingStyles.headingLight}>Sobre o Projeto</h2>
    <p style={landingStyles.textLight}>
      Este projeto utiliza inteligência artificial para interpretar dados obtidos através da Análise de Gases Dissolvidos (DGA) em óleo isolante de transformadores. O objetivo é diagnosticar de forma preventiva possíveis falhas, aumentando a confiabilidade e a segurança de sistemas elétricos.
    </p>
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
    margin: 0,
    padding: 0,
    overflowX: 'hidden',
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
  sectionLight: {
    padding: '60px 40px',
    textAlign: 'center',
    backgroundColor: '#161B22',
  },
  headingLight: {
    fontSize: '36px',
    color: '#58A6FF',
    marginBottom: '20px',
  },
  textLight: {
    fontSize: '18px',
    maxWidth: '800px',
    margin: '0 auto',
    lineHeight: '1.6',
    color: '#C9D1D9',
  },
};

export default App;
