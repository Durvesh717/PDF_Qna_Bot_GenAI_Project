from collections import defaultdict
from pathlib import Path
from typing import List

import pymupdf4llm
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_upstage import UpstageDocumentParseLoader

from core.config import get_settings
from core.logger import get_logger
from generation.llm import get_vision_llm

logger = get_logger(__name__)

IMAGE_DESCRIPTION_PROMPT = """
Describe only the factual content visible in the image:

1. If decorative/non-informational: output '<---image--->'

2. For content images:
- General Images: List visible objects, text, and measurable attributes
- Charts/Infographics: State all numerical values and labels present
- Tables: Convert to markdown table format with exact data

Rules:
* Include only directly observable information
* Use original numbers and text without modification
* Avoid any interpretation or analysis
* Preserve all labels and measurements exactly as shown
"""


def pdf_to_markdown(file_path: str | Path) -> list[dict]:
    """Convert a PDF to markdown page chunks using PyMuPDF4LLM."""
    logger.info(f"Converting PDF to markdown: {file_path}")
    return pymupdf4llm.to_markdown(
        doc=str(file_path),
        page_chunks=True,
        show_progress=True,
    )


def parse_pdf_with_upstage(file_path: str | Path) -> List[Document]:
    """Parse PDF structure using Upstage Document Parse."""
    logger.info(f"Parsing PDF structure with Upstage: {file_path}")
    settings = get_settings()
    loader = UpstageDocumentParseLoader(
        str(file_path),
        split="page",
        output_format="markdown",
        base64_encoding=["figure", "chart", "table"],
    )
    return loader.load_and_split()


def describe_images(docs: List[Document]) -> List[Document]:
    """Generate text descriptions for embedded images, charts, and tables."""
    logger.info("Generating image descriptions")
    model = get_vision_llm()
    new_documents = []

    for doc in docs:
        encodings = doc.metadata.get("base64_encodings", [])
        if not encodings:
            continue

        for idx, img_base64 in enumerate(encodings):
            message = HumanMessage(
                content=[
                    {"type": "text", "text": IMAGE_DESCRIPTION_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"},
                    },
                ]
            )

            try:
                response = model.invoke([message])
                new_documents.append(
                    Document(
                        page_content=response.content,
                        metadata={
                            "page": doc.metadata.get("page", "unknown"),
                            "type": "image_description",
                            "index": idx,
                        },
                    )
                )
            except Exception as e:
                logger.error(f"Failed to describe image on page {doc.metadata.get('page')}: {e}")

    return new_documents


def merge_text_and_images(
    md_text: list[dict], image_description_docs: List[Document]
) -> List[Document]:
    """Merge markdown text and image descriptions into per-page Documents."""
    logger.info("Merging text and image descriptions by page")
    page_contents = defaultdict(list)
    page_metadata = {}

    for text_item in md_text:
        page = int(text_item["metadata"]["page"])
        page_contents[page].append(text_item["text"])
        if page not in page_metadata:
            page_metadata[page] = {
                "source": text_item["metadata"].get("file_path", "unknown"),
                "page": page,
            }

    for img_doc in image_description_docs:
        try:
            page = int(img_doc.metadata["page"])
            page_contents[page].append(img_doc.page_content)
        except (ValueError, TypeError):
            logger.warning(f"Skipping image doc with invalid page: {img_doc.metadata}")

    merged_docs = []
    for page in sorted(page_contents.keys()):
        full_content = "\n\n".join(page_contents[page])
        merged_docs.append(
            Document(page_content=full_content, metadata=page_metadata[page])
        )

    return merged_docs


def extract_content_from_pdf(file_path: str | Path) -> List[Document] | None:
    """Full ingestion pipeline: markdown + Upstage parsing + image descriptions."""
    try:
        logger.info(f"Starting content extraction for {file_path}")
        md_text = pdf_to_markdown(file_path)
        docs = parse_pdf_with_upstage(file_path)
        image_description_docs = describe_images(docs)
        merged_docs = merge_text_and_images(md_text, image_description_docs)
        logger.info(f"Extracted {len(merged_docs)} page documents")
        return merged_docs
    except Exception as e:
        logger.exception(f"Error extracting content from PDF: {e}")
        return None
