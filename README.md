# Asistente Académico RAG CUN

Aplicación Streamlit con tres módulos: Chat RAG, generación descargable del póster académico y administración del corpus `datos_aca`.

## Arquitectura
Pregunta → Query Rewriting con Gemini 3.5 Flash → MiniLM → Qdrant Cloud → Top-k y umbral → contexto → Gemini → respuesta con fuentes.

## Corpus
Incluya en `datos_aca/` los archivos institucionales `.pdf`, `.docx` y `.txt`. Al iniciar, la aplicación crea la colección `aca_documentos_academicos` si no existe o está vacía. La pestaña **Corpus datos_aca** permite reconstruirla explícitamente cuando cambie el corpus.

## Despliegue
1. Suba a GitHub `app.py`, `requirements.txt`, `README.md`, `.gitignore`, `.streamlit/secrets.example.toml` y la carpeta `datos_aca/`.
2. En Streamlit Community Cloud seleccione `app.py`.
3. Configure en **App settings > Secrets**:
```toml
QDRANT_URL = "https://TU_CLUSTER.qdrant.io"
QDRANT_API_KEY = "..."
GEMINI_API_KEY = "..."
GEMINI_MODEL = "gemini-3.5-flash"
```
4. Verifique la cantidad de vectores, ejecute una consulta y genere el póster desde su pestaña.

## Seguridad
No publique `.streamlit/secrets.toml`. Revise que los documentos incluidos en `datos_aca` estén autorizados para el repositorio seleccionado.
