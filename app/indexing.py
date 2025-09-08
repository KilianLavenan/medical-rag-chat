import base64
import os
import re
from pathlib import Path
from typing import Any

import chromadb
from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)


def summarize_table(
    table: Table,
    has_header_row: bool = True,
    has_header_column: bool = False,
) -> str:
    """
    Convert a Word table into a structured text summary.

    The function reads the content of a `docx.table.Table`, extracts its cells,
    and generates a human-readable narrative depending on whether the table
    contains a header row, a header column, both, or neither.

    Args:
        table (Table): A `python-docx` Table object to summarize.
        has_header_row (bool, optional): If True, the first row is treated as a header. Defaults to True.
        has_header_column (bool, optional): If True, the first column is treated as a header. Defaults to False.

    Returns:
        str: A textual summary of the table content, formatted with bullet points
        and optional headers based on the provided arguments.
    """
    rows = list(table.rows)
    if not rows:
        return ""

    # Construire la matrice des cellules
    matrix = [[cell.text.strip() for cell in row.cells] for row in rows]

    narrative = []

    # Case 1 : Row header only
    if has_header_row and not has_header_column:
        headers = matrix[0]
        for row in matrix[1:]:
            if len(set(row)) == 1 and row[0]:
                narrative.append(f"Note: {row[0]}")
            else:
                items = [f"{h}: {c}" for h, c in zip(headers, row, strict=False) if c]
                if items:
                    narrative.append("- " + ", ".join(items))

    # Caes 2 : Column header only
    elif has_header_column and not has_header_row:
        for row in matrix:
            header = row[0]
            values = row[1:]
            items = [c for c in values if c]
            if items:
                narrative.append(f"- {header}: {', '.join(items)}")

    # Casee 3 : Row header + column header
    elif has_header_row and has_header_column:
        col_headers = matrix[0][1:]  # ignore (0,0)
        for row in matrix[1:]:
            row_header = row[0]
            values = row[1:]
            for j, cell in enumerate(values):
                col_header = col_headers[j]
                narrative.append(f"- {col_header} et {row_header}: {cell}")

    # Case 4 : No headers
    else:
        for row in matrix:
            narrative.append("- " + ", ".join(c for c in row if c))

    return "\n".join(narrative)


def extract_image_from_docx(docx_path: Path, output_path: Path) -> None:
    doc = Document(docx_path)
    for rel in doc.part.rels.values():
        if "image" in rel.target_ref:
            image_data = rel.target_part.blob
            with open(output_path, "wb") as img_file:
                img_file.write(image_data)


def summarize_image(image_path: Path) -> str:
    with open(image_path, "rb") as f:
        img_bytes = f.read()
    img_b64 = base64.b64encode(img_bytes).decode("utf-8")

    prompt = (
        "Analyse attentivement cette image. "
        "C'est un organigramme de prise en charge en ambulatoire ou à l'hôpital des pneumonies communautaires.\n\n"
        "Ta réponse doit suivre EXACTEMENT ce format :\n"
        "- Description : une phrase\n"
        "- Titre : le titre principal du schéma\n"
        "- Structure du schéma : décris en utilisant le texte exact dans chaque bloc du schéma, le processus de prise en charge d'un patient atteint d'une pneumonie communautaire.\n"
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Tu es un assistant médical qui décrit des schémas."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                ],
            },
        ],
    )

    return response.choices[0].message.content  # type: ignore


def chunk_paragraph_and_tables(docx_path: Path) -> list[str]:
    """
    Split a DOCX document into text chunks by paragraphs and tables.

    The function iterates through each section of the Word document, extracting
    paragraphs and converting tables into narrative form. Sections starting
    with a pattern like "A-", "B-", etc. trigger the start of a new chunk.

    Args:
        docx_path (Path): Path to the DOCX file to process.

    Returns:
        list[str]: A list of text chunks where each chunk may contain
        paragraphs and/or table narratives.
    """
    doc = Document(docx_path)
    has_header = [
        (False, True),
        (False, False),
        (True, False),
        (True, False),
        (True, True),
        (True, False),
        (True, False),
    ]
    chunks = []
    table_index = 0
    for section in doc.sections:
        current_chunk = ""
        for element in section.iter_inner_content():
            section_pattern = re.compile(r"^[A-Z]-.*")
            if isinstance(element, Table):
                current_chunk += summarize_table(element, *has_header[table_index]) + "\n"
                table_index += 1
            elif isinstance(element, Paragraph):
                text = element.text
                if section_pattern.match(text):
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                current_chunk += text.strip() + "\n"
        chunks.append(current_chunk.strip())
    return chunks


def chunk_document(docx_path: Path, image_path: Path) -> list[str]:
    """
    Split a DOCX document into text chunks and append an image summary.

    Args:
        docx_path (Path): Path to the DOCX file to process.
        image_path (Path): Path where the extracted image will be stored and
            then summarized.

    Returns:
        list[str]: A list of text chunks with the image description appended
        to the final chunk.
    """
    chunks = chunk_paragraph_and_tables(docx_path)
    if not image_path.exists():
        extract_image_from_docx(docx_path, image_path)
    image_summary = summarize_image(image_path)
    chunks[-1] += "\n" + image_summary
    return chunks


def create_embeddings(chunks: list[str]) -> list[dict[str, Any]]:
    data_with_embeddings = []
    embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    for chunk in chunks:
        chunk_embedding = embedding_model.encode(chunk).tolist()
        data_with_embeddings.append({"chunk": chunk, "embedding": chunk_embedding})
    return data_with_embeddings


def store_embeddings_in_chroma(data_with_embeddings: list[dict[str, Any]], chroma_db_path: Path) -> None:
    client = chromadb.PersistentClient(path=chroma_db_path)
    collection = client.get_or_create_collection(name="medical_rag")
    for i, item in enumerate(data_with_embeddings):
        collection.add(
            ids=[f"chunk_{i}"],
            embeddings=[item["embedding"]],
            documents=[item["chunk"]],
        )


def main() -> None:
    """
    Orchestrate the full document processing and indexing pipeline.

    This function extracts text and image content from the DOCX file,
    generates embeddings for each chunk, and stores them in a persistent
    ChromaDB database for later retrieval.
    """
    docx_path = Path("data/Prise en charge des Pneumopathies aigues communautaires V2.docx")
    image_path = Path("data/extracted_image.png")
    chroma_db_path = Path("data/chroma_db")
    if not chroma_db_path.exists():
        chunks = chunk_document(docx_path, image_path)
        data_with_embeddings = create_embeddings(chunks)
        store_embeddings_in_chroma(data_with_embeddings, chroma_db_path)
