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
    modelos = list(dict.fromkeys([GEMINI_MODEL, "gemini-3.5-flash", "gemini-1.5-flash"]))
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
            if any(code in str(e) for code in ["404", "NOT_FOUND", "503", "UNAVAILABLE"]):
                continue
            raise e
    raise RuntimeError(f"No se pudo generar respuesta: {ultimo_error}")


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
        # Si falla por cuota (429), devuelve la consulta original sin interrumpir la búsqueda
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
        if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
            top = hits[0]
            response = (
                f"⚠️ **Aviso de cuota API (Error 429):** Se alcanzó el límite diario gratuito de Gemini. "
                f"Sin embargo, la recuperación semántica en Qdrant encontró evidencia relevante en "
                f"**[{top.get('filename')}, p. {top.get('page')}]** (score: {top['score']:.4f}):\n\n"
                f"> {top.get('text', '')}"
            )
        else:
            raise e
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


def build_poster(config, last_result):
    prs = Presentation()
    prs.slide_width = Inches(16.54)
    prs.slide_height = Inches(11.69)
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = RGBColor(241, 246, 250)
    navy = (14, 39, 66)
    blue = (26, 115, 232)
    white = (255, 255, 255)
    dark = (34, 45, 57)

    header = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, Inches(1.05))
    header.fill.solid()
    header.fill.fore_color.rgb = RGBColor(*navy)
    header.line.color.rgb = RGBColor(*navy)
    add_text(slide, .4, .14, 12, .42, config["title"], 24, True, white)
    add_text(slide, .42, .61, 12, .26, "Integrantes: " + config["members"], 10, False, white)

    sections = [
        (.35, 1.3, 4.05, 1.35, "SITUACIÓN PROBLEMA", config["problem"]),
        (.35, 2.82, 4.05, 1.35, "OBJETIVO GENERAL", config["objective"]),
        (.35, 4.34, 4.05, .95, "PILAR SELECCIONADO", config["pillar"]),
        (.35, 5.46, 4.05, 2.25, "CONCLUSIONES", config["conclusions"]),
    ]
    for x, y, w, h, t, b in sections:
        add_box(slide, x, y, w, h)
        add_text(slide, x + .16, y + .1, w - .32, .28, t, 11, True, blue)
        add_text(slide, x + .16, y + .43, w - .32, h - .5, b, 9, False, dark)

    add_box(slide, 4.62, 1.3, 7.1, 4.0)
    add_text(slide, 4.8, 1.43, 6.75, .3, "ARQUITECTURA RAG COMPLETA", 13, True, blue, PP_ALIGN.CENTER)
    labels = [
        "Pregunta",
        "Query Rewriting\nGemini 3.5",
        "MiniLM\nEmbeddings",
        "Qdrant Cloud",
        "Top-k + umbral",
        "Gemini 3.5\nRespuesta",
        "Streamlit\nUI + fuentes",
    ]
    xs = [4.85, 5.8, 6.8, 7.8, 8.82, 9.86, 10.87]
    for i, (label, x) in enumerate(zip(labels, xs)):
        add_box(slide, x, 2.15, .78, 2.1, blue, blue)
        add_text(slide, x + .03, 2.35, .72, 1.55, label, 7.5, True, white, PP_ALIGN.CENTER)
        if i < len(xs) - 1:
            connector = slide.shapes.add_connector(
                MSO_CONNECTOR.STRAIGHT, Inches(x + .78), Inches(3.2), Inches(xs[i + 1]), Inches(3.2)
            )
            connector.line.color.rgb = RGBColor(*navy)

    add_text(
        slide,
        4.9,
        4.55,
        6.5,
        .42,
        "Documentos → chunks → MiniLM → Qdrant | Consulta → recuperación → contexto → Gemini → UI",
        8.5,
        False,
        dark,
        PP_ALIGN.CENTER,
    )
    add_box(slide, 4.62, 5.46, 7.1, 3.25)
    add_text(slide, 4.8, 5.58, 6.75, .3, "CAPTURAS REALES DE STREAMLIT", 12, True, blue, PP_ALIGN.CENTER)
    for i, (path, label) in enumerate([(config["shot1"], "Consulta"), (config["shot2"], "Respuesta + fuentes")]):
        x = 4.9 + i * 3.45
        if path is not None:
            slide.shapes.add_picture(path, Inches(x), Inches(6.05), Inches(3.05), Inches(2.2))
        else:
            add_box(slide, x, 6.05, 3.05, 2.2, (235, 239, 244), (170, 185, 200))
            add_text(slide, x + .2, 6.85, 2.65, .45, "Captura real pendiente", 10, True, (100, 116, 139), PP_ALIGN.CENTER)
        add_text(slide, x, 8.3, 3.05, .22, label, 8, True, navy, PP_ALIGN.CENTER)

    add_box(slide, 11.95, 1.3, 4.2, 5.0)
    add_text(slide, 12.12, 1.45, 3.85, .3, "EJEMPLO DE INTERACCIÓN", 12, True, blue)
    if last_result:
        answer_text = last_result["answer"][:800]
        sources = "\n".join(
            f'• {h.get("filename")}, p. {h.get("page")}, score {h["score"]:.3f}'
            for h in last_result["hits"][:3]
        )
        question = last_result.get("question", "")
    else:
        question = "Ejecute una consulta en Chat RAG antes de generar el póster."
        answer_text = "Respuesta real pendiente."
        sources = "Fuentes reales pendientes."
    add_text(slide, 12.12, 1.92, 3.85, .65, "Pregunta:\n" + question, 9, True, navy)
    add_text(slide, 12.12, 2.72, 3.85, 1.72, "Respuesta:\n" + answer_text, 8.5, False, dark)
    add_text(slide, 12.12, 4.56, 3.85, 1.4, "Citas:\n" + sources, 8, False, dark)

    add_box(slide, 11.95, 6.48, 4.2, 2.23)
    add_text(slide, 12.12, 6.62, 3.85, .28, "ENLACES FUNCIONALES", 12, True, blue, PP_ALIGN.CENTER)
    for i, (label, url) in enumerate([("GitHub", config["github"]), ("App", config["app_url"]), ("Video", config["video"])]):
        x = 12.18 + i * 1.42
        if url:
            qr = qrcode.make(url)
            bio = io.BytesIO()
            qr.save(bio, format="PNG")
            bio.seek(0)
            slide.shapes.add_picture(bio, Inches(x), Inches(7.05), Inches(.75), Inches(.75))
        else:
            add_box(slide, x, 7.05, .75, .75, (235, 239, 244), (180, 190, 200))
            add_text(slide, x, 7.25, .75, .22, "PEND.", 7, True, (100, 116, 139), PP_ALIGN.CENTER)
        add_text(slide, x - .1, 7.84, .95, .2, label, 8, True, navy, PP_ALIGN.CENTER)

    footer = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(.35), Inches(9.05), Inches(15.8), Inches(1.9))
    footer.fill.solid()
    footer.fill.fore_color.rgb = RGBColor(*navy)
    footer.line.color.rgb = RGBColor(*navy)
    add_text(
        slide,
        .58,
        9.3,
        15.3,
        .34,
        "TRAZABILIDAD: documento • página • chunk_id • score | Secrets fuera de GitHub | datos_aca incluido",
        10,
        True,
        white,
        PP_ALIGN.CENTER,
    )
    add_text(
        slide,
        .58,
        9.82,
        15.3,
        .6,
        "Colección aca_documentos_academicos | MiniLM multilingual | Qdrant Cloud | Gemini 3.5 Flash | Streamlit",
        9,
        False,
        white,
        PP_ALIGN.CENTER,
    )
    output = io.BytesIO()
    prs.save(output)
    output.seek(0)
    return output.getvalue()


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
    st.subheader("Generar póster académico editable")
    c1, c2 = st.columns(2)
    with c1:
        title = st.text_input("Título", "ASISTENTE ACADÉMICO RAG EN LA NUBE")
        members = st.text_input("Integrantes", "Diego Fernando López Ledezma")
        problem = st.text_area(
            "Situación problema",
            "La consulta manual de reglamentos dispersa información y dificulta entregar respuestas verificables.",
        )
        objective = st.text_area(
            "Objetivo general",
            "Desarrollar un asistente académico RAG con Qdrant Cloud y Gemini para producir respuestas trazables.",
        )
        pillar = st.text_input("Pilar seleccionado", "Innovación educativa apoyada en IA y cloud")
        conclusions = st.text_area(
            "Conclusiones",
            "• RAG separa recuperación y generación.\n• Qdrant conserva vectores y metadatos.\n• Las citas mejoran la trazabilidad.",
        )
    with c2:
        github = st.text_input("URL de GitHub")
        app_url = st.text_input("URL de Streamlit")
        video = st.text_input("URL del video")
        shot1 = st.file_uploader("Captura 1: consulta", type=["png", "jpg", "jpeg"])
        shot2 = st.file_uploader("Captura 2: respuesta y fuentes", type=["png", "jpg", "jpeg"])
    config = {
        "title": title,
        "members": members,
        "problem": problem,
        "objective": objective,
        "pillar": pillar,
        "conclusions": conclusions,
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
