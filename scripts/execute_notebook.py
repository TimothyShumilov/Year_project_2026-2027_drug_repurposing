"""Execute every EDA cell using this interpreter, preserving completed outputs."""
import os
import sys
from pathlib import Path
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager

ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
os.environ.setdefault('IPYTHONDIR',str(ROOT/'.cache/ipython'))
os.environ.setdefault('JUPYTER_RUNTIME_DIR',str(ROOT/'.cache/jupyter'))
for name in ['matplotlib','ipython','jupyter']:
    (ROOT/'.cache'/name).mkdir(parents=True,exist_ok=True)
manager=KernelManager(kernel_name='python3')
manager.kernel_spec.argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}']
path=ROOT/'EDA.ipynb'
notebook=nbformat.read(path,as_version=4)
client=NotebookClient(notebook,km=manager,timeout=14400,resources={'metadata':{'path':str(ROOT)}})
try:
    client.execute()
finally:
    nbformat.write(notebook,path)
    if manager.has_kernel:
        manager.shutdown_kernel(now=True)
print(f'Executed {sum(c.cell_type=="code" for c in notebook.cells)} code cells: {path}')
