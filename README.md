# Proyecto BREIT - ONG TANI Desarrollo Infantil G1

Este repositorio contiene el código y documentación para el análisis de datos y generación de reportes utilizando Python y Quarto. El proyecto está estructurado para facilitar la colaboración y el desarrollo reproducible.

## 🛠️ Tecnologías Utilizadas

- **Python** - Lenguaje principal
- **Quarto** - Documentación y reportes reproducibles
- **UV** - Gestor de dependencias moderno y rápido
- **Virtual Environment** - Aislamiento de dependencias

## 📋 Requisitos Previos

### 1. Python
Asegúrate de tener Python 3.8+ instalado:
```bash
python --version
```

### 2. UV (Recomendado)
Instala UV como gestor de dependencias:
```bash
# En macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# En Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# O usando pip
pip install uv
```

### 3. Quarto
Instala Quarto desde su [página oficial](https://quarto.org/docs/get-started/):

**macOS:**
```bash
brew install quarto
```

**Windows:**
Descarga el instalador desde [quarto.org](https://quarto.org/docs/get-started/)

**Linux:**
```bash
# Ubuntu/Debian
sudo curl -LO https://quarto.org/download/latest/quarto-linux-amd64.deb
sudo gdebi quarto-linux-amd64.deb
```

**RECOMENDADO: INSTALAR EXTENSION DE QUARTO EN VSCODE!!!**

## 🚀 Configuración del Proyecto

### Opción A: Colaborador del Proyecto (Push directo)

### 1. Clonar el Repositorio
```bash
git clone https://github.com/tu-usuario/tu-repositorio.git
cd tu-repositorio
```

### Opción B: Contribuidor Externo (Fork + Pull Request)

Para contribuir al proyecto sin ser colaborador directo, sigue estos pasos:

### 1. Fork del Repositorio
1. Ve al repositorio principal en GitHub
2. Haz clic en el botón **"Fork"** en la esquina superior derecha
3. Esto creará una copia del repositorio en tu cuenta de GitHub

📺 **Video tutorial detallado:** [Cómo hacer fork en GitHub](https://www.youtube.com/watch?v=a_FLqX3vGR4)

### 2. Clonar tu Fork
```bash
# Clona TU fork (no el repositorio original)
git clone https://github.com/TU-USUARIO/NOMBRE-PROYECTO.git
cd NOMBRE-PROYECTO
```

### 3. Configurar Remotes
```bash
# Agregar el repositorio original como "upstream"
git remote add upstream https://github.com/USUARIO-ORIGINAL/NOMBRE-PROYECTO.git

# Verificar remotos
git remote -v
# origin    https://github.com/TU-USUARIO/NOMBRE-PROYECTO.git (fetch)
# origin    https://github.com/TU-USUARIO/NOMBRE-PROYECTO.git (push)
# upstream  https://github.com/USUARIO-ORIGINAL/NOMBRE-PROYECTO.git (fetch)
# upstream  https://github.com/USUARIO-ORIGINAL/NOMBRE-PROYECTO.git (push)
```

### 4. Configuración Inicial (Ambas opciones)

### 2. Crear Entorno Virtual con UV
```bash
# Crear entorno virtual usando UV
uv venv

# Activar entorno virtual
# En Windows
.venv\Scripts\activate

# En macOS/Linux
source .venv/bin/activate
```

### 3. Instalar Dependencias con UV
```bash
# Instalar todas las dependencias desde pyproject.toml
uv sync

# O alternativamente
uv pip install -e .
```

## 📁 Estructura del Proyecto

```
TANI_PROJECT_BREIT/
├── .venv/                              # Entorno virtual (creado con uv)
├── Data/                               # Datos del proyecto
│   ├── assets/                         # Recursos adicionales
│   ├── RUTAS.md                        # Documentación de fuentes
│   ├── diccionario_X_detalle.qmd      # Diccionario de datos
├── EDAs/                              # Scripts de análisis exploratorio
│   ├── EDA_*.qmd                      # Análisis exploratorio de datos
├── pyproject.toml                     # Configuración y dependencias
├── uv.lock                            # Lock file de UV (si existe)
└── README.md                          # Este archivo
```

## 🔧 USO DEL PROYECTO

### Ejecutar Scripts Python
```bash
# Activar entorno virtual primero
source .venv/bin/activate  # Linux/macOS
# o
.venv\Scripts\activate     # Windows

# Ejecutar scripts específicos
python maquina_bolas_data_generator.py
python verificar_target.py
```

### Renderizar Documentos Quarto
```bash
# Renderizar un documento específico
quarto render EDA_maquina_bolas.qmd

# Renderizar todos los documentos .qmd
quarto render

# Previsualizar en tiempo real
quarto preview EDA_maquina_bolas.qmd
```

### Trabajar con pyproject.toml y UV
```bash
# Sincronizar todas las dependencias del proyecto
uv sync

# Agregar nueva dependencia
uv add nombre-paquete

# Agregar dependencia de desarrollo
uv add --dev nombre-paquete-dev

# Actualizar dependencias
uv lock --upgrade

# Instalar proyecto en modo desarrollo
uv pip install -e .

# Generar requirements.txt si es necesario
uv pip freeze > requirements.txt
```

## 📊 Flujo de Trabajo Típico

1. **Activar el entorno virtual:**
   ```bash
   source .venv/bin/activate  # Linux/macOS
   # o
   .venv\Scripts\activate     # Windows
   ```
2. **Sincronizar dependencias:**
   ```bash
   uv sync
   ```
3. **Ejecutar archivos quarto (render):**
   ```bash
   quarto render EDAs/EDA_*.qmd
   ```

## 🔀 Flujo de Trabajo con Git

### Para Colaboradores Directos:

```bash
# 1. Asegurarse de estar en main/master actualizado
git checkout main
git pull origin main

# 2. Crear rama para nueva feature
git checkout -b feature/nombre-descriptivo

# 3. Trabajar en tus cambios...
# ... hacer modificaciones ...

# 4. Añadir y commitear cambios
git add .
git commit -m "Descripción clara de los cambios"

# 5. Subir rama al repositorio
git push origin feature/nombre-descriptivo

# 6. Crear Pull Request en GitHub
# Ve a GitHub y crea el PR desde tu rama hacia main
```

### Para Contribuidores Externos (Fork workflow):

```bash
# 1. Mantener tu fork actualizado
git checkout main
git fetch upstream
git merge upstream/main
git push origin main

# 2. Crear rama para nueva feature
git checkout -b feature/nombre-descriptivo

# 3. Trabajar en tus cambios...
# ... hacer modificaciones ...

# 4. Añadir y commitear cambios
git add .
git commit -m "Descripción clara de los cambios"

# 5. Subir cambios a TU fork
git push origin feature/nombre-descriptivo

# 6. Crear Pull Request
# Ve a GitHub, navega a tu fork
# GitHub mostrará un botón para crear PR hacia el repo original
```

### Comandos Git Útiles:

```bash
# Ver estado de archivos
git status

# Ver diferencias antes de commit
git diff

# Ver historial de commits
git log --oneline

# Cambiar entre ramas
git checkout nombre-rama

# Ver todas las ramas
git branch -a

# Eliminar rama local (después de merge)
git branch -d feature/nombre-rama

# Actualizar desde el repositorio original (solo para forks)
git fetch upstream
git merge upstream/main
```


## 🔍 Análisis Disponibles

- **EDA_maquina_bolas.qmd** - Análisis exploratorio de datos

## 🤝 Contribución

1. Fork el proyecto
2. Crea tu rama de feature (`git checkout -b feature/NuevaCaracteristica`)
3. Commit tus cambios (`git commit -m 'Agregar nueva característica'`)
4. Push a la rama (`git push origin feature/NuevaCaracteristica`)
5. Abre un Pull Request

## ⚠️ Solución de Problemas

### Error de Quarto no encontrado
```bash
# Verificar instalación
quarto --version

# Reinstalar si es necesario
# Seguir instrucciones de instalación arriba
```

### Problemas con UV
```bash
# Verificar instalación
uv --version

# Reinstalar
pip install --upgrade uv
```

### Problemas con Git
```bash
# Conflictos de merge
git status  # Ver archivos en conflicto
# Editar archivos para resolver conflictos
git add .
git commit -m "Resolver conflictos de merge"

# Deshacer último commit (mantener cambios)
git reset --soft HEAD~1

# Deshacer cambios no commiteados
git checkout -- archivo.py
git reset --hard HEAD  # ¡CUIDADO! Elimina todos los cambios
```

### Entorno virtual no se activa
```bash
# Recrear entorno virtual con UV
rm -rf .venv
uv venv
source .venv/bin/activate  # Linux/macOS
```

## 📄 Licencia

MIT License

## 👨‍💻 Autores

1. Angel Ch
2. Jherson  
3. Alan
4. Edson

---

**Nota:** Recuerda siempre activar tu entorno virtual antes de trabajar en el proyecto y mantener tus dependencias actualizadas.