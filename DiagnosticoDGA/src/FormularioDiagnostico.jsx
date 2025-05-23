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
      <div style={styles.formDiv}>
        <form style={styles.formulario} onSubmit={diagnosticar}>
          <div style={styles.divTitulo}>
            <h2 style={styles.titulo}>
              Diagnóstico de<br /> transformadores de<br /> potência por DGA
            </h2>
          </div>

          <div style={styles.conjunto}>
            <div style={styles.campo}>
              <label style={styles.label}>H2 (ppm):</label>
              <input
                type="number"
                name="H2"
                value={inputs.H2}
                onChange={handleChange}
                required
                style={styles.input}
              />
            </div>

            <div style={styles.campo}>
              <label style={styles.label}>CH4 (ppm):</label>
              <input
                type="number"
                name="CH4"
                value={inputs.CH4}
                onChange={handleChange}
                required
                style={styles.input}
              />
            </div>
          </div>

          <div style={styles.conjunto}>
            <div style={styles.campo}>
              <label style={styles.label}>C2H2 (ppm):</label>
              <input
                type="number"
                name="C2H2"
                value={inputs.C2H2}
                onChange={handleChange}
                required
                style={styles.input}
              />
            </div>

            <div style={styles.campo}>
              <label style={styles.label}>C2H4 (ppm):</label>
              <input
                type="number"
                name="C2H4"
                value={inputs.C2H4}
                onChange={handleChange}
                required
                style={styles.input}
              />
            </div>
          </div>

          <div style={styles.sozinho}>
            <div style={styles.campo}>
              <label style={styles.label}>C2H6 (ppm):</label>
              <input
                type="number"
                name="C2H6"
                value={inputs.C2H6}
                onChange={handleChange}
                required
                style={styles.inputSolo}
              />
            </div>
          </div>

          <div style={styles.divBotao}>
            <button type="submit" style={styles.botao}>Confirmar</button>
          </div>
        </form>
      </div>
      
      {resultado && (
        <div style={styles.formDiv}>
          <div style={styles.formulario}>
            <h4 style={styles.titulo}>Falha identificada:</h4>
            <p style={styles.titulo}><strong>{resultado.falha}</strong></p>
          </div>
        </div>
      )}
    </div>
  );
};

const styles = {
  container: {
    width: '100vw',
    height: '100vh',
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    padding: '20px',
    backgroundColor: '#0D1117',   
  },

  formDiv: {
    marginTop: '20px',
    background: 'linear-gradient(90deg, #0F2027 0%, #203A43 50%, #2C5364 100%)',
    padding: '3px',
    borderRadius: '12px',
    width: '100%',
    maxWidth: '450px',
    boxShadow: '0 4px 12px rgba(0,0,0,0.4)',
  },

  formulario: {
    alignItems: 'center',
    backgroundColor: '#161B22',
    borderRadius: '12px',
    padding: '30px',
    display: 'flex',
    flexDirection: 'column',
    gap: '15px',
    fontFamily: 'Courier New, monospace',
  },

  divTitulo: {
    marginBottom: '25px',
    textAlign: 'center',
    borderBottom: '1px solid #30363D',
    paddingBottom: '10px',
  },

  titulo: {
    color: '#C9D1D9',
    fontSize: '30px',
    fontWeight: 'bold',
    lineHeight: '1.3',
  },

  conjunto: {
    display: 'flex',
    alignItems: 'center',
    gap: '10px'
  },

  sozinho: {
    display: 'flex',
    justifyContent: 'center',
  },

  campo: {
    display: 'flex',
    flexDirection: 'column',

  },

  label: {
    color: '#8B949E',
    marginBottom: '5px',
    paddingLeft: '5px',
    fontSize: '14px',
    fontWeight: '500',
  },

  input: {
    width: '175px',
    backgroundColor: '#0D1117',
    color: '#C9D1D9',
    border: '1px solid #30363D',
    borderRadius: '8px',
    padding: '5px',
    paddingLeft: '10px',
    fontSize: '16px',
    outline: 'none',
    transition: 'border-color 0.3s',
  },

  inputSolo: {
    width: '360px',
    backgroundColor: '#0D1117',
    color: '#C9D1D9',
    border: '1px solid #30363D',
    borderRadius: '8px',
    padding: '5px',
    paddingLeft: '10px',
    fontSize: '16px',
    outline: 'none',
    transition: 'border-color 0.3s',
  },

  divBotao: {
    display: 'flex',
    justifyContent: 'center',
    marginTop: '20px',
  },

  botao: {
    width: '350px',
    backgroundColor: '#238636',
    color: '#ffffff',
    padding: '12px 30px',
    border: 'none',
    borderRadius: '5px',
    fontSize: '18px',
    fontWeight: 'bold',
    cursor: 'pointer',
    transition: 'background-color 0.3s ease',
  },

  resultado: {
    marginTop: '30px',
    backgroundColor: '#1F6FEB',
    color: 'white',
    borderRadius: '8px',
    padding: '20px',
    textAlign: 'center',
    fontWeight: 'bold',
    fontSize: '18px',
  },
};

export default FormularioDiagnostico;
