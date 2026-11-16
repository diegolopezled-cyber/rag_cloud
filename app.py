import io
import time
import uuid
import hashlib
from pathlib import Path

import fitz
import qrcode
import streamlit as st
from docx import Document
from google import genai
from google.genai import types
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="Asistente académico CUN", page_icon="📘", layout="wide")
COLLECTION = "aca_documentos_academicos"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DATA_DIR = Path(__file__).parent / "datos_aca"
GEMINI_MODEL = st.secrets.get("GEMINI_MODEL", "gemini-3.5-flash")
MENSAJE_SIN_CONTEXTO = "No encuentro información suficiente en los documentos consultados para responder con seguridad."


def required_secret(name):
    value = st.secrets.get(name)
    if not value:
        st.error(f"Falta configurar el secreto {name} en Streamlit Community Cloud.")
        st.stop()
    return value


@st.cache_resource(show_spinner="Cargando modelos y conexión cloud...")
def resources():
    emb = SentenceTransformer(EMBEDDING_MODEL)
    qd = QdrantClient(
        url=required_secret("QDRANT_URL"),
        api_key=required_secret("QDRANT_API_KEY"),
        timeout=60,
    )
    llm = genai.Client(api_key=required_secret("GEMINI_API_KEY"))
    return emb, qd, llm


embedding_model, qdrant, llm = resources()


def clean_text(text):
    return " ".join((text or "").replace("\x00", " ").split())


def extract_documents(folder):
    docs = []
    if not folder.exists():
        return docs
    for path in sorted(folder.rglob("*")):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        try:
            if suffix == ".pdf":
                pdf = fitz.open(path)
                for page_number, page in enumerate(pdf, 1):
                    text = clean_text(page.get_text("text"))
                    if text:
                        docs.append({"filename": path.name, "page": page_number, "text": text})
            elif suffix == ".docx":
                doc = Document(path)
                text = clean_text("\n".join(p.text for p in doc.paragraphs))
                if text:
                    docs.append({"filename": path.name, "page": 1, "text": text})
            elif suffix == ".txt":
                text = clean_text(path.read_text(encoding="utf-8", errors="ignore"))
                if text:
                    docs.append({"filename": path.name, "page": 1, "text": text})
        except Exception as exc:
            st.warning(f"No se pudo leer {path.name}: {exc}")
    return docs


def chunk_documents(docs, chunk_size=1000, overlap=180):
    chunks = []
    step = max(1, chunk_size - overlap)
    for doc in docs:
        text = doc["text"]
        for index, start in enumerate(range(0, len(text), step)):
            fragment = text[start:start + chunk_size].strip()
            if len(fragment) < 80:
                continue
            raw_id = f'{doc["filename"]}:{doc["page"]}:{index}:{fragment[:80]}'
            chunk_id = hashlib.sha1(raw_id.encode("utf-8")).hexdigest()[:20]
            chunks.append({**doc, "text": fragment, "chunk_id": chunk_id, "chunk_index": index})
    return chunks


def collection_points():
    existing = {c.name for c in qdrant.get_collections().collections}
    if COLLECTION not in existing:
        return 0
    return int(qdrant.get_collection(COLLECTION).points_count or 0)


def index_bundled_corpus(force=False):
    docs = extract_documents(DATA_DIR)
    if not docs:
        raise FileNotFoundError("La carpeta datos_aca no contiene PDF, DOCX o TXT.")
    chunks = chunk_documents(docs)
    if not chunks:
        raise RuntimeError("No se obtuvieron fragmentos válidos del corpus.")
    vectors = embedding_model.encode(
        [c["text"] for c in chunks], normalize_embeddings=True, show_progress_bar=False
    )
    existing = {c.name for c in qdrant.get_collections().collections}
    if COLLECTION not in existing:
        qdrant.create_collection(
            collection_name=COLLECTION,
            vectors_config=models.VectorParams(size=len(vectors[0]), distance=models.Distance.COSINE),
        )
    elif force:
        qdrant.delete_collection(COLLECTION)
        qdrant.create_collection(
            collection_name=COLLECTION,
            vectors_config=models.VectorParams(size=len(vectors[0]), distance=models.Distance.COSINE),
        )
    points = []
    for chunk, vector in zip(chunks, vectors):
        pid = str(uuid.uuid5(uuid.NAMESPACE_URL, f'aca-cun:{chunk["chunk_id"]}'))
        points.append(models.PointStruct(id=pid, vector=vector.tolist(), payload=chunk))
    for start in range(0, len(points), 64):
        qdrant.upsert(COLLECTION, points=points[start:start + 64], wait=True)
    return len(points), len(docs)


def ensure_collection():
    points = collection_points()
    if points == 0 and DATA_DIR.exists():
        with st.spinner("Inicializando Qdrant con el corpus incluido en datos_aca..."):
            index_bundled_corpus(force=False)
        points = collection_points()
    return points


def generate_text(prompt, temperature=0.1, max_tokens=800):
    modelos = list(dict.fromkeys([
        GEMINI_MODEL,
        "gemini-3.5-flash",
        "gemini-3.5-flash-lite",
        "gemini-2.5-flash",
        "gemini-1.5-flash",
    ]))
    ultimo_error = None
    for modelo in modelos:
        try:
            response = llm.models.generate_content(
                model=modelo,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=temperature, max_output_tokens=max_tokens
                ),
            )
            if response.text:
                return response.text.strip()
        except Exception as e:
            ultimo_error = e
            err_str = str(e)
            if any(code in err_str for code in ["404", "NOT_FOUND", "503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED"]):
                continue
            raise e
    raise RuntimeError(f"Ningún modelo de Gemini pudo responder. Último error: {ultimo_error}")


def rewrite_query(query):
    try:
        return generate_text(
            "Reescribe esta consulta para recuperación semántica en reglamentos académicos "
            "y de bienestar de la CUN. Conserva exactamente la intención, no agregues hechos, "
            f"no respondas y devuelve solo la consulta reescrita: {query}",
            temperature=0.0,
            max_tokens=200,
        )
    except Exception:
        return query


def answer(query, k, threshold, use_rewriting=True):
    started = time.perf_counter()
    rewritten = rewrite_query(query) if use_rewriting else query
    vector = embedding_model.encode(rewritten, normalize_embeddings=True).tolist()
    points = qdrant.query_points(
        collection_name=COLLECTION,
        query=vector,
        limit=k,
        with_payload=True,
        with_vectors=False,
    ).points
    hits = []
    for point in points:
        if float(point.score) >= threshold:
            payload = point.payload or {}
            hits.append({"score": float(point.score), **payload})
    if not hits:
        return {
            "answer": MENSAJE_SIN_CONTEXTO,
            "hits": [],
            "latency": time.perf_counter() - started,
            "rewritten": rewritten,
        }
    context = "\n\n".join(
        f'Fuente {i}: Documento={h.get("filename")}; página={h.get("page")}; '
        f'chunk_id={h.get("chunk_id")}; score={h["score"]:.4f}\n{h.get("text", "")}'
        for i, h in enumerate(hits, 1)
    )
    prompt = (
        "Responde exclusivamente con el contexto. No inventes información. "
        f"Si el contexto no basta, responde exactamente: {MENSAJE_SIN_CONTEXTO} "
        "Incluye citas [Documento, p. X].\n\n"
        f"CONTEXTO:\n{context}\n\nPREGUNTA ORIGINAL:\n{query}"
    )
    try:
        response = generate_text(prompt)
    except Exception as e:
        top = hits[0]
        response = (
            f"⚠️ **Aviso del servicio generativo (Gemini API):** `{type(e).__name__}: {e}`\n\n"
            f"La recuperación semántica en **Qdrant Cloud** funcionó correctamente y encontró evidencia en "
            f"**[{top.get('filename')}, p. {top.get('page')}]** (score: `{top['score']:.4f}`):\n\n"
            f"> {top.get('text', '')}"
        )
    return {
        "answer": response,
        "hits": hits,
        "latency": time.perf_counter() - started,
        "rewritten": rewritten,
    }


def add_text(slide, x, y, w, h, value, size=10, bold=False, color=(34, 45, 57), align=PP_ALIGN.LEFT):
    shape = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    frame = shape.text_frame
    frame.clear()
    frame.word_wrap = True
    paragraph = frame.paragraphs[0]
    paragraph.text = str(value)
    paragraph.alignment = align
    for run in paragraph.runs:
        run.font.name = "Aptos"
        run.font.size = Pt(size)
        run.font.bold = bold
        run.font.color.rgb = RGBColor(*color)
    return shape


def add_box(slide, x, y, w, h, fill=(255, 255, 255), line=(205, 216, 228)):
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    shape.fill.solid()
    shape.fill.fore_color.rgb = RGBColor(*fill)
    shape.line.color.rgb = RGBColor(*line)
    return shape


def add_metric_card(slide, x, y, w, h, number, label, accent=(26, 115, 232)):
    add_box(slide, x, y, w, h, (247, 250, 253), (205, 216, 228))
    add_text(slide, x + .08, y + .08, w - .16, .35, number, 18, True, accent, PP_ALIGN.CENTER)
    add_text(slide, x + .08, y + .48, w - .16, .28, label, 7.5, True, (55, 65, 81), PP_ALIGN.CENTER)


def build_poster(config, last_result):
    prs = Presentation()
    prs.slide_width = Inches(16.54)
    prs.slide_height = Inches(11.69)
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = RGBColor(241, 246, 250)
    navy = (14, 39, 66); blue = (26, 115, 232); cyan = (0, 153, 188)
    green = (25, 135, 84); amber = (221, 126, 15); white = (255, 255, 255); dark = (34, 45, 57)

    header = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, Inches(1.02))
    header.fill.solid(); header.fill.fore_color.rgb = RGBColor(*navy); header.line.color.rgb = RGBColor(*navy)
    add_text(slide, .38, .10, 12.7, .41, config['title'], 23, True, white)
    add_text(slide, .40, .57, 12.8, .24, 'Autor: ' + config['members'] + ' | Especialización en Inteligencia Artificial - CUN', 9.3, False, white)
    add_text(slide, 13.35, .22, 2.75, .42, 'ACA 1 + ACA 2', 13, True, (145, 205, 255), PP_ALIGN.CENTER)

    # Left column: problem, objective and methodology.
    left_sections = [
        (1.22, 1.48, 'SITUACIÓN PROBLEMA', config['problem']),
        (2.86, 1.42, 'OBJETIVO GENERAL', config['objective']),
        (4.44, 1.62, 'METODOLOGÍA', config['methodology']),
        (6.22, 1.22, 'CORPUS Y PREPARACIÓN', config['corpus']),
    ]
    for y, h, title, body in left_sections:
        add_box(slide, .34, y, 4.02, h)
        add_text(slide, .50, y + .10, 3.70, .25, title, 10.5, True, blue)
        add_text(slide, .50, y + .40, 3.70, h - .48, body, 8.15, False, dark)

    # Center top: editable native architecture.
    add_box(slide, 4.57, 1.22, 7.34, 3.16)
    add_text(slide, 4.77, 1.34, 6.94, .28, 'ARQUITECTURA RAG Y TRAZABILIDAD', 12.5, True, blue, PP_ALIGN.CENTER)
    labels = ['Documentos\nPDF', 'Limpieza +\nchunks', 'MiniLM\nembeddings', 'Qdrant\nCloud', 'Top-k +\numbral', 'Gemini API\ncontexto', 'Respuesta +\nfuentes']
    xs = [4.78, 5.77, 6.76, 7.75, 8.74, 9.73, 10.72]
    colors=[navy, cyan, blue, navy, green, amber, blue]
    for i,(label,x,color) in enumerate(zip(labels,xs,colors)):
        add_box(slide, x, 1.92, .82, 1.46, color, color)
        add_text(slide, x+.04, 2.16, .74, .88, label, 7.4, True, white, PP_ALIGN.CENTER)
        if i < len(xs)-1:
            con=slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x+.82), Inches(2.66), Inches(xs[i+1]), Inches(2.66))
            con.line.color.rgb=RGBColor(*navy); con.line.width=Pt(1.8)
    add_text(slide, 4.91, 3.57, 6.65, .45, 'Cada fragmento conserva documento, página, chunk_id y score. El LLM recibe únicamente el contexto recuperado, no el corpus completo.', 8.3, False, dark, PP_ALIGN.CENTER)

    # Center middle: data cards.
    add_text(slide, 4.72, 4.55, 7.02, .27, 'CONFIGURACIÓN Y DATOS DEL EXPERIMENTO', 11.5, True, blue, PP_ALIGN.CENTER)
    metrics=[('255','vectores en Qdrant'),('3','documentos PDF'),('Top-5','recuperación'),('0,45','score mínimo'),('1000','caracteres/chunk'),('150','solapamiento')]
    for i,(num,label) in enumerate(metrics):
        row=i//3; col=i%3
        add_metric_card(slide, 4.67+col*2.39, 4.91+row*.95, 2.18, .82, num, label, [blue,cyan,green,amber,blue,cyan][i])

    # Center lower: results and evaluation.
    add_box(slide, 4.57, 6.92, 3.55, 2.38)
    add_text(slide, 4.75, 7.05, 3.20, .27, 'RESULTADOS DE RECUPERACIÓN', 10.5, True, blue, PP_ALIGN.CENTER)
    retrieval = ('Consulta de homologación, Top-5:\n'
                 '1. 0,703 | Reglamento estudiantil, p.11\n'
                 '2. 0,697 | Reglamento de bienestar, p.24\n'
                 '3. 0,693 | Reglamento estudiantil, p.15\n'
                 '4. 0,686 | Reglamento estudiantil, p.14\n'
                 '5. 0,681 | Reglamento estudiantil, p.11')
    add_text(slide, 4.78, 7.43, 3.13, 1.53, retrieval, 7.7, False, dark)

    add_box(slide, 8.35, 6.92, 3.56, 2.38)
    add_text(slide, 8.53, 7.05, 3.20, .27, 'QUERY REWRITING Y EVALUACIÓN', 10.5, True, blue, PP_ALIGN.CENTER)
    eval_text=('“tutorías”: Top-1 0,608 → 0,749\nMejora observada: +0,141 puntos (~23 %).\n\n'
               'Batería: 6 casos (directos, multi-contexto, fuera del corpus y seguridad).\n'
               'Nota: las métricas de latencia del respaldo son simuladas por cuota 429 y no equivalen a un benchmark real.')
    add_text(slide, 8.57, 7.43, 3.12, 1.58, eval_text, 7.7, False, dark)

    # Right column: controls, evidence and conclusions.
    add_box(slide, 12.12, 1.22, 4.08, 2.30)
    add_text(slide, 12.30, 1.35, 3.72, .27, 'CONTROLES DE FIDELIDAD Y SEGURIDAD', 10.5, True, blue, PP_ALIGN.CENTER)
    controls=('• Recuperación antes de generación.\n• Filtro MIN_SCORE = 0,45.\n• Abstención cuando el contexto es insuficiente.\n• Citas con documento y página.\n• Secrets fuera del notebook y GitHub.\n• Revisión humana frente a la fuente original.')
    add_text(slide, 12.32, 1.77, 3.68, 1.49, controls, 8.1, False, dark)

    add_box(slide, 12.12, 3.72, 4.08, 2.62)
    add_text(slide, 12.30, 3.85, 3.72, .27, 'EVIDENCIA DE STREAMLIT', 10.5, True, blue, PP_ALIGN.CENTER)
    if config['shot1'] is not None:
        slide.shapes.add_picture(config['shot1'], Inches(12.35), Inches(4.27), Inches(1.65), Inches(1.40))
    else:
        add_box(slide, 12.35, 4.27, 1.65, 1.40, (235,239,244), (170,185,200)); add_text(slide, 12.47, 4.80, 1.4, .28, 'Consulta\npendiente', 8, True, (100,116,139), PP_ALIGN.CENTER)
    if config['shot2'] is not None:
        slide.shapes.add_picture(config['shot2'], Inches(14.25), Inches(4.27), Inches(1.65), Inches(1.40))
    else:
        add_box(slide, 14.25, 4.27, 1.65, 1.40, (235,239,244), (170,185,200)); add_text(slide, 14.37, 4.80, 1.4, .28, 'Fuentes\npendientes', 8, True, (100,116,139), PP_ALIGN.CENTER)
    add_text(slide, 12.36, 5.80, 3.55, .27, 'La interfaz permite inspeccionar fragmentos y metadatos recuperados.', 7.7, False, dark, PP_ALIGN.CENTER)

    add_box(slide, 12.12, 6.55, 4.08, 2.75)
    add_text(slide, 12.30, 6.68, 3.72, .27, 'CONCLUSIONES Y LIMITACIONES', 10.5, True, blue, PP_ALIGN.CENTER)
    add_text(slide, 12.32, 7.08, 3.68, 1.86, config['conclusions'] + '\n\nLimitaciones: ' + config['limitations'], 7.65, False, dark)

    # Footer with links and concise technology trace.
    footer=slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(.34), Inches(9.58), Inches(15.86), Inches(1.60))
    footer.fill.solid(); footer.fill.fore_color.rgb=RGBColor(*navy); footer.line.color.rgb=RGBColor(*navy)
    add_text(slide, .56, 9.76, 10.7, .35, 'TRAZABILIDAD: documento • página • chunk_id • score | Corpus autorizado | Secrets protegidos', 9.3, True, white, PP_ALIGN.CENTER)
    add_text(slide, .56, 10.26, 10.7, .43, 'MiniLM multilingüe | Qdrant Cloud | Gemini API | Streamlit | Python', 8.7, False, white, PP_ALIGN.CENTER)
    for i,(label,url) in enumerate([('GitHub',config['github']),('App',config['app_url']),('Video',config['video'])]):
        x=11.52+i*1.47
        if url:
            qr=qrcode.make(url); bio=io.BytesIO(); qr.save(bio,format='PNG'); bio.seek(0)
            slide.shapes.add_picture(bio, Inches(x), Inches(9.77), Inches(.73), Inches(.73))
        else:
            add_box(slide,x,9.77,.73,.73,(235,239,244),(180,190,200)); add_text(slide,x,10.02,.73,.2,'PEND.',6.5,True,(100,116,139),PP_ALIGN.CENTER)
        add_text(slide, x-.08, 10.55, .90, .20, label, 7.4, True, white, PP_ALIGN.CENTER)

    # Speaker note equivalent is intentionally omitted; this is a one-slide poster.
    output=io.BytesIO(); prs.save(output); output.seek(0); return output.getvalue()


points = ensure_collection()
st.title("📘 Asistente académico RAG CUN")
st.caption("ACA 1 + ACA 2 · Qdrant Cloud · Gemini 3.5 Flash · Streamlit")
chat_tab, poster_tab, corpus_tab = st.tabs(["💬 Chat RAG", "🎨 Póster / Infografía", "📚 Corpus datos_aca"])

with st.sidebar:
    st.metric("Vectores en Qdrant", points)
    st.caption(f"Modelo: {GEMINI_MODEL}")
    top_k = st.slider("Top-k", 1, 10, 5)
    threshold = st.slider("Score mínimo", 0.0, 1.0, .45, .01)
    use_rewriting = st.checkbox("Activar Query Rewriting", value=True, help="Desactívalo para ahorrar 50% de cuota de API.")

with chat_tab:
    query = st.text_input("Consulta", placeholder="Ej.: ¿Cuáles son los requisitos para homologar una asignatura?")
    if st.button("Consultar", type="primary", disabled=not query):
        with st.spinner("Consultando el corpus..."):
            result = answer(query, top_k, threshold, use_rewriting=use_rewriting)
            result["question"] = query
            st.session_state["last_result"] = result
        st.subheader("Respuesta")
        st.write(result["answer"])
        st.caption(f'Consulta usada: {result["rewritten"]} | Latencia: {result["latency"]:.2f} s')
        st.subheader("Fuentes recuperadas")
        for i, hit in enumerate(result["hits"], 1):
            with st.expander(f'{i}. {hit.get("filename")} · p. {hit.get("page")} · score {hit["score"]:.4f}'):
                st.write(hit.get("text", ""))
                st.code(f'chunk_id={hit.get("chunk_id")}')

with poster_tab:
    st.subheader("Generar póster académico ampliado y editable")
    c1, c2 = st.columns(2)
    with c1:
        title = st.text_input("Título", "ASISTENTE ACADÉMICO RAG EN LA NUBE")
        members = st.text_input("Integrantes", "Diego Fernando López Ledezma")
        problem = st.text_area(
            "Situación problema",
            "La consulta manual de reglamentos académicos y documentos de bienestar distribuye la información entre distintas fuentes, exige localizar e interpretar múltiples secciones y dificulta verificar el origen de cada respuesta. El asistente RAG recupera evidencia institucional antes de generar y permite mostrar documento, página, fragmento y score.",
            height=150,
        )
        objective = st.text_area(
            "Objetivo general",
            "Desarrollar y evaluar un asistente académico en la nube basado en RAG que reutilice la base vectorial del ACA 1 e integre recuperación semántica, Qdrant Cloud, Gemini y Streamlit para generar respuestas fundamentadas, trazables y con mecanismos de abstención.",
            height=140,
        )
        methodology = st.text_area(
            "Metodología",
            "Se aplicó un flujo preparar → recuperar → generar → evaluar → presentar. Los documentos se limpiaron, dividieron en chunks, representaron con MiniLM e indexaron en Qdrant. Cada consulta utiliza Top-k, umbral mínimo y generación restringida al contexto recuperado.",
            height=140,
        )
        corpus = st.text_area(
            "Corpus y preparación",
            "Corpus: reglamento estudiantil, reglamento de bienestar y política de bienestar CUN. Se verificaron duplicados mediante SHA-256. Configuración experimental: 255 vectores, chunks de 1.000 caracteres y solapamiento de 150.",
            height=120,
        )
        pillar = st.text_input("Pilar seleccionado", "Innovación educativa apoyada en IA y cloud")
        conclusions = st.text_area(
            "Conclusiones",
            "• RAG separa recuperación y generación.\n• Qdrant conserva vectores y metadatos.\n• Las citas mejoran la trazabilidad.",
        )
    with c2:
        github = st.text_input("URL de GitHub", "https://github.com/diegolopezled-cyber/rag_cloud.git")
        app_url = st.text_input("URL de Streamlit", "")
        video = st.text_input("URL del video", "https://drive.google.com/file/d/1-N9m7Ab9aLAH1q9xWeOcPYdGYT7kvA-l/view?usp=sharing")
        shot1 = st.file_uploader("Captura 1: consulta", type=["png", "jpg", "jpeg"])
        shot2 = st.file_uploader("Captura 2: respuesta y fuentes", type=["png", "jpg", "jpeg"])
    config = {
        "title": title,
        "members": members,
        "problem": problem,
        "objective": objective,
        "methodology": methodology,
        "corpus": corpus,
        "pillar": pillar,
        "conclusions": conclusions,
        "limitations": """La muestra de evaluación es reducida; la fidelidad normativa requiere revisión humana; el sistema depende de servicios cloud y cuotas de API; un score alto no garantiza por sí solo una respuesta correcta.""",
        "github": github,
        "app_url": app_url,
        "video": video,
        "shot1": shot1,
        "shot2": shot2,
    }
    if st.button("Generar póster PPTX", type="primary"):
        poster = build_poster(config, st.session_state.get("last_result"))
        st.session_state["poster_bytes"] = poster
    if "poster_bytes" in st.session_state:
        st.download_button(
            "⬇️ Descargar Poster_Infografia_RAG_CUN.pptx",
            st.session_state["poster_bytes"],
            "Poster_Infografia_RAG_CUN.pptx",
            "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        )

with corpus_tab:
    st.subheader("Documentos incluidos en datos_aca")
    files = [p for p in DATA_DIR.rglob("*") if p.is_file()] if DATA_DIR.exists() else []
    st.write(f"Archivos incluidos en el repositorio: **{len(files)}**")
    for p in files:
        st.write("•", p.relative_to(DATA_DIR))
    st.warning("Reindexar elimina y reconstruye la colección. Úselo solo cuando cambie el corpus.")
    if st.button("Reindexar Qdrant desde datos_aca"):
        with st.spinner("Generando embeddings e indexando..."):
            count, docs = index_bundled_corpus(force=True)
        st.success(f"Indexación completada: {count} fragmentos de {docs} unidades documentales.")
        st.cache_resource.clear()
        st.rerun()
