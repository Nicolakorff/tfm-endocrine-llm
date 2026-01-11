# Guía de Instalación

**Sistema de Neuromodulación Endocrina para LLMs**  
**Versión:** 0.6.0  
**Última actualización:** Enero 2026

---

## Tabla de Contenidos

1. [Requisitos del Sistema](#requisitos-del-sistema)
2. [Instalación Rápida](#instalación-rápida)
3. [Instalación Detallada](#instalación-detallada)
4. [Instalación desde Código Fuente](#instalación-desde-código-fuente)
5. [Instalación en Diferentes Entornos](#instalación-en-diferentes-entornos)
6. [Verificación de la Instalación](#verificación-de-la-instalación)
7. [Configuración Opcional](#configuración-opcional)
8. [Troubleshooting](#troubleshooting)
9. [Desinstalación](#desinstalación)

---

## Requisitos del Sistema

### Sistema Operativo

- **Linux** (Ubuntu 20.04+, Debian 10+, CentOS 7+)
- **macOS** (10.15 Catalina o superior)
- **Windows** (10/11 con WSL2 recomendado)

### Software Base

| Componente | Versión Mínima | Versión Recomendada |
|------------|----------------|---------------------|
| **Python** | 3.8 | 3.10 o 3.11 |
| **pip** | 20.0 | Última versión |
| **Git** | 2.20 | Última versión |

### Hardware

#### Requisitos Mínimos

- **CPU:** 2 núcleos
- **RAM:** 4 GB
- **Disco:** 2 GB libres
- **GPU:** Opcional (CPU funciona)

#### Configuración Recomendada

- **CPU:** 4+ núcleos
- **RAM:** 8 GB
- **Disco:** 5 GB libres (para modelos y datos)
- **GPU:** NVIDIA con CUDA 11.7+ (para mejor rendimiento)

---

## Instalación Rápida

### Opción 1: Instalación Básica

Para usar el sistema **sin características semánticas**:

```bash
pip install git+https://github.com/Nicolakorff/tfm-endocrine-llm.git
```

**Tiempo estimado:** ~2-3 minutos

---

### Opción 2: Instalación con Sesgos Semánticos

Para incluir **sentence-transformers** (sesgos basados en embeddings):

```bash
pip install "git+https://github.com/Nicolakorff/tfm-endocrine-llm.git#egg=endocrine-llm[semantic]"
```

**Tiempo estimado:** ~5-7 minutos (descarga modelo SBERT ~80MB)

---

### Opción 3: Instalación Completa

Para desarrollo con todas las herramientas:

```bash
pip install "git+https://github.com/Nicolakorff/tfm-endocrine-llm.git#egg=endocrine-llm[all]"
```

**Incluye:** semantic + testing + docs + notebooks

**Tiempo estimado:** ~8-10 minutos

---

## Instalación Detallada

### Paso 1: Verificar Python

```bash
# Verificar versión de Python
python --version
# o
python3 --version

# Debe mostrar: Python 3.8.x o superior
```

**Si Python no está instalado:**

<details>
<summary>Linux (Ubuntu/Debian)</summary>

```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv
```
</details>

<details>
<summary>macOS</summary>

```bash
# Opción 1: Homebrew (recomendado)
brew install python@3.11

# Opción 2: Descargar desde python.org
# https://www.python.org/downloads/macos/
```
</details>

<details>
<summary>Windows</summary>

```powershell
# Descargar instalador desde:
# https://www.python.org/downloads/windows/

# Durante instalación, marcar:
# ☑ Add Python to PATH
# ☑ Install pip
```
</details>

---

### Paso 2: Crear Entorno Virtual (Recomendado)

```bash
# Crear entorno virtual
python -m venv endocrine-env

# Activar entorno
# En Linux/macOS:
source endocrine-env/bin/activate

# En Windows:
endocrine-env\Scripts\activate

# Verificar activación (debe mostrar (endocrine-env) en prompt)
```

**Ventajas del entorno virtual:**
- Aislamiento de dependencias
- Sin conflictos con otros proyectos
- Fácil de eliminar completamente

---

### Paso 3: Actualizar pip

```bash
pip install --upgrade pip setuptools wheel
```

---

### Paso 4: Instalar endocrine-llm

**Opción A: Instalación Básica**

```bash
pip install git+https://github.com/Nicolakorff/tfm-endocrine-llm.git
```

**Opción B: Con Características Semánticas**

```bash
pip install "git+https://github.com/Nicolakorff/tfm-endocrine-llm.git#egg=endocrine-llm[semantic]"
```

**Opción C: Instalación de Desarrollo**

```bash
pip install "git+https://github.com/Nicolakorff/tfm-endocrine-llm.git#egg=endocrine-llm[all]"
```

---

### Paso 5: Verificar Instalación

```python
# En terminal Python
python

# Ejecutar:
>>> import endocrine_llm
>>> endocrine_llm.__version__
'0.5.0'

>>> from endocrine_llm import EndocrineModulatedLLM
>>> # Si no hay errores, instalación exitosa
```

---

## Instalación desde Código Fuente

### Para Desarrolladores o Contribuidores

#### Paso 1: Clonar Repositorio

```bash
git clone https://github.com/Nicolakorff/tfm-endocrine-llm.git
cd tfm-endocrine-llm
```

#### Paso 2: Crear Entorno Virtual

```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
# o
venv\Scripts\activate  # Windows
```

#### Paso 3: Instalación en Modo Editable

**Instalación básica:**
```bash
pip install -e .
```

**Con características semánticas:**
```bash
pip install -e ".[semantic]"
```

**Instalación completa de desarrollo:**
```bash
pip install -e ".[all]"
```

**Incluye:**
- Todas las dependencias de producción
- Herramientas de testing (pytest, pytest-cov)
- Linters y formatters (black, isort, flake8)
- Documentación (sphinx)
- Jupyter notebooks

---

#### Paso 4: Verificar Instalación Editable

```bash
# Hacer un cambio en el código
echo "# Test comment" >> endocrine_llm/core.py

# Importar sin reinstalar
python -c "import endocrine_llm; print('✓ Editable install works')"

# Revertir cambio
git checkout endocrine_llm/core.py
```

---

## Instalación en Diferentes Entornos

### Google Colab

```python
# En una celda de Colab
!pip install git+https://github.com/Nicolakorff/tfm-endocrine-llm.git

# Con sesgos semánticos
!pip install "git+https://github.com/Nicolakorff/tfm-endocrine-llm.git#egg=endocrine-llm[semantic]"

# Reiniciar runtime si es necesario
import os
os.kill(os.getpid(), 9)
```

---

### Jupyter Notebook (Local)

```bash
# Activar entorno virtual primero
source endocrine-env/bin/activate

# Instalar paquete
pip install git+https://github.com/Nicolakorff/tfm-endocrine-llm.git

# Instalar kernel de Jupyter
pip install ipykernel
python -m ipykernel install --user --name=endocrine-env

# Iniciar Jupyter
jupyter notebook
```

**En el notebook, seleccionar kernel:** `endocrine-env`

---

### Docker

```dockerfile
# Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y git

# Instalar endocrine-llm
RUN pip install git+https://github.com/Nicolakorff/tfm-endocrine-llm.git

# Comando por defecto
CMD ["python"]
```

**Construir y ejecutar:**

```bash
# Construir imagen
docker build -t endocrine-llm .

# Ejecutar contenedor
docker run -it endocrine-llm python

# Dentro del contenedor:
>>> from endocrine_llm import EndocrineModulatedLLM
>>> model = EndocrineModulatedLLM("distilgpt2")
```

---

### Conda (Anaconda/Miniconda)

```bash
# Crear entorno conda
conda create -n endocrine python=3.10

# Activar entorno
conda activate endocrine

# Instalar con pip (dentro del entorno conda)
pip install git+https://github.com/Nicolakorff/tfm-endocrine-llm.git
```

**Nota:** Usar `pip` dentro de conda está soportado oficialmente.

---

## Verificación de la Instalación

### Test Básico

```python
from endocrine_llm import EndocrineModulatedLLM, HORMONE_PROFILES

# Inicializar modelo
model = EndocrineModulatedLLM("distilgpt2")

# Generar texto
texts = model.generate_with_hormones(
    prompt="Hello, I am",
    hormone_profile=HORMONE_PROFILES["baseline"],
    max_new_tokens=20
)

print("Instalación exitosa:")
print(texts[0])
```

**Salida esperada:**
```
Instalación exitosa:
Hello, I am a software engineer working on machine learning projects...
```

---

### Test con Características Semánticas

```python
# Solo si instalaste [semantic]
from endocrine_llm.semantic import SemanticBiasManager

try:
    manager = SemanticBiasManager(model.tokenizer, device=model.device)
    print("✓ Características semánticas disponibles")
except ImportError:
    print("✗ Sentence-transformers no instalado")
    print("  Instalar con: pip install sentence-transformers")
```

---

### Test Completo

```bash
# Si instalaste [all], ejecutar tests
pytest tests/ -v

# Con cobertura
pytest tests/ --cov=endocrine_llm --cov-report=html
```

---

## Configuración Opcional

### GPU (CUDA)

**Verificar disponibilidad de CUDA:**

```python
import torch
print(f"CUDA disponible: {torch.cuda.is_available()}")
print(f"Versión CUDA: {torch.version.cuda}")
print(f"Dispositivo: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
```

**Si CUDA no está disponible pero tienes GPU NVIDIA:**

1. Instalar CUDA Toolkit: https://developer.nvidia.com/cuda-downloads
2. Reinstalar PyTorch con soporte CUDA:

```bash
# Para CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Para CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

### Configurar Device por Defecto

```python
# En tu código
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Usar GPU 0

# O forzar CPU
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # No usar GPU

from endocrine_llm import EndocrineModulatedLLM
model = EndocrineModulatedLLM("gpt2", device="cpu")  # Explícito
```

---

### Descargar Modelos Previamente

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Descargar modelo una vez
model_name = "distilgpt2"
AutoModelForCausalLM.from_pretrained(model_name)
AutoTokenizer.from_pretrained(model_name)

print(f"Modelo {model_name} descargado")
```

**Ubicación del cache:** `~/.cache/huggingface/hub/`

---

## Troubleshooting

### Problema 1: ModuleNotFoundError

**Error:**
```
ModuleNotFoundError: No module named 'endocrine_llm'
```

**Solución:**
```bash
# Verificar instalación
pip list | grep endocrine

# Si no aparece, reinstalar
pip install git+https://github.com/Nicolakorff/tfm-endocrine-llm.git

# Verificar que Python usa el entorno correcto
which python  # Linux/macOS
where python  # Windows
```

---

### Problema 2: ImportError con sentence-transformers

**Error:**
```
ImportError: cannot import name 'SemanticBiasManager'
ModuleNotFoundError: No module named 'sentence_transformers'
```

**Solución:**
```bash
pip install sentence-transformers
# O reinstalar con [semantic]
pip install "endocrine-llm[semantic]"
```

---

### Problema 3: CUDA out of memory

**Error:**
```
RuntimeError: CUDA out of memory
```

**Soluciones:**

```python
# Opción 1: Usar modelo más pequeño
model = EndocrineModulatedLLM("distilgpt2")  # En lugar de "gpt2-large"

# Opción 2: Forzar CPU
model = EndocrineModulatedLLM("gpt2", device="cpu")

# Opción 3: Reducir batch size
texts = model.generate_with_hormones(
    prompt,
    profile,
    num_return_sequences=1  # En lugar de 5
)
```

---

### Problema 4: Versión de Python incompatible

**Error:**
```
ERROR: Package requires Python >=3.8
```

**Solución:**

```bash
# Verificar versión instalada
python --version

# Instalar Python 3.10 (recomendado)
# Linux:
sudo apt install python3.10 python3.10-venv

# macOS:
brew install python@3.10

# Windows: Descargar desde python.org
```

---

### Problema 5: Conflictos de dependencias

**Error:**
```
ERROR: pip's dependency resolver does not currently take into account all the packages...
```

**Solución:**

```bash
# Crear entorno limpio
python -m venv fresh-env
source fresh-env/bin/activate  # Linux/macOS
fresh-env\Scripts\activate  # Windows

# Instalar en orden
pip install --upgrade pip
pip install git+https://github.com/Nicolakorff/tfm-endocrine-llm.git
```

---

### Problema 6: SSL Certificate Error

**Error:**
```
SSL: CERTIFICATE_VERIFY_FAILED
```

**Solución:**

```bash
# Opción 1: Actualizar certificados (macOS)
/Applications/Python\ 3.10/Install\ Certificates.command

# Opción 2: Usar HTTP (temporal, no recomendado)
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org \
    git+https://github.com/Nicolakorff/tfm-endocrine-llm.git
```

---

### Problema 7: Git no instalado

**Error:**
```
fatal: 'git' is not recognized as an internal or external command
```

**Solución:**

```bash
# Linux (Ubuntu/Debian)
sudo apt install git

# macOS
brew install git

# Windows: Descargar Git Bash
# https://git-scm.com/download/win
```

---

## Desinstalación

### Desinstalar Paquete

```bash
pip uninstall endocrine-llm
```

### Eliminar Entorno Virtual Completo

```bash
# Desactivar entorno
deactivate

# Eliminar directorio
rm -rf endocrine-env  # Linux/macOS
rmdir /s endocrine-env  # Windows
```

### Limpiar Cache de HuggingFace

```bash
# Eliminar modelos descargados
rm -rf ~/.cache/huggingface/
```

**Espacio liberado:** ~500MB - 2GB dependiendo de modelos descargados

---

## Instalación para Casos Específicos

### Instalación Sin Internet (Offline)

**Paso 1:** En máquina con internet, descargar dependencias:

```bash
pip download endocrine-llm -d ./packages/
```

**Paso 2:** Transferir carpeta `packages/` a máquina offline

**Paso 3:** Instalar en máquina offline:

```bash
pip install --no-index --find-links ./packages/ endocrine-llm
```

---

### Instalación con Versión Específica

```bash
# Instalar versión específica
pip install git+https://github.com/Nicolakorff/tfm-endocrine-llm.git@v0.5.0

# Instalar desde commit específico
pip install git+https://github.com/Nicolakorff/tfm-endocrine-llm.git@abc1234
```

---

### Instalación en Servidor sin Privilegios Root

```bash
# Instalar en directorio de usuario
pip install --user git+https://github.com/Nicolakorff/tfm-endocrine-llm.git

# Verificar instalación
python -m endocrine_llm --version
```

---

## Resumen de Comandos

### Instalación Rápida

```bash
# Básico
pip install git+https://github.com/Nicolakorff/tfm-endocrine-llm.git

# Con sesgos semánticos
pip install "git+https://github.com/Nicolakorff/tfm-endocrine-llm.git#egg=endocrine-llm[semantic]"

# Completo
pip install "git+https://github.com/Nicolakorff/tfm-endocrine-llm.git#egg=endocrine-llm[all]"
```

### Desde Código Fuente

```bash
git clone https://github.com/Nicolakorff/tfm-endocrine-llm.git
cd tfm-endocrine-llm
pip install -e ".[all]"
```

### Verificación

```python
python -c "from endocrine_llm import EndocrineModulatedLLM; print('✓ OK')"
```
---

**Última actualización:** Enero 2026  
**Versión del sistema:** 0.6.0

**FIN DE DOCUMENTO**
