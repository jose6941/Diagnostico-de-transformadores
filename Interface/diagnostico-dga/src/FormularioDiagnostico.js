import React, { useState } from 'react';

const FormularioDiagnostico = () => {
  const [inputs, setInputs] = useState({ H2: '', CH4: '', C2H2: '', C2H4: '', C2H6: '' });
  const [resultado, setResultado] = useState(null);

  const handleChange = e => {
    setInputs({ ...inputs, [e.target.name]: e.target.value });
  };

  const diagnosticar = async e => {
    e.preventDefault();
    const res = await fetch('http://localhost:8000/diagnostico', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        H2: parseFloat(inputs.H2),
        CH4: parseFloat(inputs.CH4),
        C2H2: parseFloat(inputs.C2H2),
        C2H4: parseFloat(inputs.C2H4),
        C2H6: parseFloat(inputs.C2H6)
      })
    });
    const data = await res.json();
    setResultado(data);
  };

  return (
    <div style={styles.container}>
      
      <form style={styles.formulario} onSubmit={diagnosticar}>
        <h2 style={styles.titulo}>Diagnóstico DGA</h2>
        {['H2', 'CH4', 'C2H2', 'C2H4', 'C2H6'].map(gas => (
          <div key={gas} style={styles.campo}>
            <label style={styles.label}>{gas} (ppm):</label>
            <div style={styles.divInput}>
              <input
                type="number"
                name={gas}
                value={inputs[gas]}
                onChange={handleChange}
                required
                style={styles.input}
              />
            </div>
            
          </div>
        ))}
        <button type="submit" style={styles.botao}>Diagnosticar</button>
      </form>

      {resultado && (
        <div style={styles.resultado}>
          <h4>Falha identificada:</h4>
          <p><strong>{resultado.falha}</strong></p>
        </div>
      )}
    </div>
  );
};

const styles = {
  container: {
    width: '100vw',
    heigh: '100vh',
    padding: '3rem',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
  },
  titulo: {
    textAlign: 'center',
    color: 'black',
    fontSize: 30,
    marginBottom: '2rem'
  },
  formulario: {
    display: 'flex',
    justifyContent: 'center',
    alignItems: 'center',
    flexDirection: 'column',
    padding: '10px 40px',
    backgroundColor: 'white',
    borderRadius: '8px',
    boxShadow: '2px 2px 8px black'
  },
  campo: {
    marginBottom: '1.2rem'
  },
  label: {
    marginBottom: '3px',
    display: 'block',
    color: '#333',
    fontWeight: 'bold'
  },
  input: {
    width: '220px',
    borderRadius: '3px',
    paddingBottom: '6px',
    border: 'none',
    borderBottom: '1px solid #ccc',
    fontSize: '15px',
    alignItems: 'center',
    outline: 'none'
  },
  divInput: {
    width: '220px',
    border: 'none',
    borderBottom: '1px solid #ccc',
  },
  botao: {
    backgroundColor: '#0077aa',
    width: 200,
    color: '#fff',
    paddingTop: '15px',
    paddingBottom: '15px',
    border: 'none',
    borderRadius: '5px',
    fontSize: '1rem',
    cursor: 'pointer',
    marginTop: '10px',
    marginBottom: '10px'
  },
  resultado: {
    marginTop: '2rem',
    background: '#e6f7ff',
    padding: '1.2rem',
    borderLeft: '6px solid #00aaff',
    borderRadius: '10px'
  }
};

export default FormularioDiagnostico;
