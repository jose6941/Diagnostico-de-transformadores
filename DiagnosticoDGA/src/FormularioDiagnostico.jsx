import React, { useState } from 'react';
import './App.css';

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
            <h2 style={styles.titulo}>Diagnóstico de<br/> transformadores de<br/>  potência por DGA</h2>
          </div>
          
          <div style={styles.conjunto}>
            <div style={styles.campo}>
              <label style={styles.label}>H2 (ppm):</label>
              <div style={styles.divInput}>
                <input
                  type="number"
                  name={'H2'}
                  value={inputs['H2']}
                  onChange={handleChange}
                  required
                  style={styles.input}
                />
              </div>
            </div>

            <div style={styles.campo}>
              <label style={styles.label}>CH4 (ppm):</label>
              <div style={styles.divInput}>
                <input
                  type="number"
                  name={'CH4'}
                  value={inputs['CH4']}
                  onChange={handleChange}
                  required
                  style={styles.input}
                />
              </div>
            </div>
          </div>

           
          <div style={styles.conjunto}> 
              <div style={styles.campo}>
                <label style={styles.label}>C2H2 (ppm):</label>
                <input
                  type="number"
                  name={'C2H2'}
                  value={inputs['C2H2']}
                  onChange={handleChange}
                  required
                  style={styles.input}
                />
              </div>
          
              <div style={styles.campo}>
                <label style={styles.label}>C2H4 (ppm):</label>
                <input
                  type="number"
                  name={'C2H4'}
                  value={inputs['C2H4']}
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
                name={'C2H6'}
                value={inputs['C2H6']}
                onChange={handleChange}
                required
                style={styles.input}
              />
            </div>
          </div>

          <div style={styles.divBotao}>
              <button type="submit" style={styles.botao}>Confirmar</button>
          </div>
          
        </form>
      </div>
      
      <div style={styles.formDiv}>
        {resultado && (
        <div style={styles.formulario}>
          <h4 style={styles.titulo}>Falha identificada:</h4>
          <p style={styles.titulo}><strong>{resultado.falha}</strong></p>
        </div>
        )}
      </div>
      
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
    boxSizing: 'borderBox',
    backgroundColor: '#151320',
  },
  titulo: {
    font: 'bold',
    textAlign: 'center',
    color: '#151320',
    fontSize: 28,
  },
  divTitulo: {
    borderBottom: '1px solid #151320',
    marginBottom: '2rem',
    marginTop: '1.5rem'
  },
  formDiv: {
    marginTop: '20px',
    backgroundImage: 'linear-gradient(90deg, #9572FC 0%, #43E7AD 50.52%, #E2D45C 100%)',
    padding: '5px',
    borderRadius: '15px',
  },
  formulario: {
    fontFamily: 'Monospace',
    backgroundColor: 'white',
    paddingTop: '10px',
    paddingLeft: '40px',
    paddingRight: '40px',
    borderRadius: '15px',
    alignItems: 'center'
  },
  conjunto: {
    display: 'flex',
    flex: 'row',
  },
  sozinho: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
  },
  campo: {
    margin: '0 auto',
    borderRadius: '5px',
    marginBottom: '15px',
    borderBottom: '1px solid #151320'
  },
  label: {
    display: 'block',
    color: '#151320',
    fontSize: '15px',
    fontWeight: 'bold'
  },
  input: {
    width: '110px',
    height: '15px',
    borderRadius: '10px',
    padding: '6px',
    border: 'none',
    fontSize: '15px',
    outline: 'none'
  },
  divBotao: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
  },
  botao: {
    backgroundColor: '#151320',
    width: '300px',
    height: '50px',
    color: '#fff',
    border: 'none',
    borderRadius: '6px',
    fontSize: '20px',
    cursor: 'pointer',
    marginTop: '25px',
    marginBottom: '10px',
  },
  resultado: {
    display: 'flex',
    marginTop: '2rem',
    background: '#e6f7ff',
    padding: '1.2rem',
    borderLeft: '6px solid #151320',
    borderRadius: '10px'
  }
};

export default FormularioDiagnostico;
