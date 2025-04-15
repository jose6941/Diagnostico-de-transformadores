import subprocess

subprocess.run('pip install -r requirements.txt')
subprocess.run('py Pre_processamento/DataCleaning.py')
subprocess.run('py Pre_processamento/DataNormalization.py')
1