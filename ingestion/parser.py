import base64
import io
from collections import defaultdict
from pathlib import Path

import pymupdf4llm
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from PIL import Image

from core.config import Settings, get_settings
from core.logger import get_logger
from generation.llm import get_vision_llm

logger = get_logger(__name__)

IMAGE_DESCRIPTION_PROMPT = """
Describe only the factual content visible in the image:

1. If decorative/non-informational: output '<---image--->'

2. For content images:
- General Images: list visible objects, text, and measurable attributes
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


def extract_images_from_pdf(
    file_path: str | Path, settings: Settings | None = None
) -> list[Document]:
    """Extract images from each page and describe them using the vision LLM."""
    logger.info(f"Extracting images from PDF: {file_path}")
    import fitz  # PyMuPDF

    settings = settings or get_settings()
    doc = fitz.open(str(file_path))
    model = None
    image_docs = []

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        images = page.get_images(full=True)

        for img_index, img in enumerate(images, start=1):
            # Cap total described images: each is a paid vision-LLM call, so an
            # image-stuffed PDF is a cost-amplification vector.
            if len(image_docs) >= settings.max_images_per_pdf:
                logger.warning(
                    f"Reached image cap ({settings.max_images_per_pdf}); "
                    "skipping remaining images"
                )
                doc.close()
                return image_docs

            xref = img[0]
            try:
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]

                # Skip small or likely decorative images
                if len(image_bytes) < 2048:
                    continue

                # Lazy-load vision model only when an image is found
                if model is None:
                    try:
                        model = get_vision_llm(settings.vision_provider, settings.vision_model)
                    except Exception as e:
                        logger.warning(f"Vision model unavailable, skipping image descriptions: {e}")
                        doc.close()
                        return image_docs

                # Convert to JPEG for consistency
                pil_image = Image.open(io.BytesIO(image_bytes))
                if pil_image.mode in ("RGBA", "P"):
                    pil_image = pil_image.convert("RGB")
                buffered = io.BytesIO()
                pil_image.save(buffered, format="JPEG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

                message = HumanMessage(
                    content=[
                        {"type": "text", "text": IMAGE_DESCRIPTION_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"},
                        },
                    ]
                )

                response = model.invoke([message])
                image_docs.append(
                    Document(
                        page_content=response.content,
                        metadata={
                            "page": page_num + 1,
                            "type": "image_description",
                            "index": img_index,
                            "source": str(file_path),
                        },
                    )
                )
            except Exception as e:
                logger.warning(f"Failed to process image on page {page_num + 1}: {e}")

    doc.close()
    return image_docs


def merge_text_and_images(
    md_text: list[dict], image_description_docs: list[Document]
) -> list[Document]:
    """Merge markdown text and image descriptions into per-page Documents."""
    logger.info("Merging text and image descriptions by page")
    page_contents = defaultdict(list)
    page_metadata = {}

    for text_item in md_text:
        page = int(text_item["metadata"].get("page", text_item["metadata"].get("page_number", 1)))
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


def extract_content_from_pdf(
    file_path: str | Path, settings: Settings | None = None
) -> list[Document] | None:
    """Full ingestion pipeline: markdown text + image descriptions."""
    try:
        logger.info(f"Starting content extraction for {file_path}")
        md_text = pdf_to_markdown(file_path)
        image_description_docs = extract_images_from_pdf(file_path, settings)
        merged_docs = merge_text_and_images(md_text, image_description_docs)
        logger.info(f"Extracted {len(merged_docs)} page documents")
        return merged_docs
    except Exception as e:
        logger.exception(f"Error extracting content from PDF: {e}")
        return None
