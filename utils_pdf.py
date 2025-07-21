import pdfplumber
from PIL import Image
import pytesseract
import pandas as pd
import streamlit as st
import logging

def extract_text_tables_images_from_pdf(pdf_path):
    """
    Extracts text, tables (as markdown), and images (with OCR) from a PDF in reading order.
    Returns a single string with all content in order.
    """
    content = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                try:
                    # Extract text
                    text = page.extract_text() or ""
                    if text.strip():
                        content.append(text.strip())
                except Exception as e:
                    st.error(f"❌ Error extracting text from page {page_num+1}: {e}")
                    logging.error(f"Error extracting text from page {page_num+1}: {e}")
                try:
                    # Extract tables
                    tables = page.extract_tables()
                    for table in tables:
                        if table and any(any(cell for cell in row) for row in table):
                            df = pd.DataFrame(table)
                            md_table = df.to_markdown(index=False)
                            content.append(md_table)
                except Exception as e:
                    st.error(f"❌ Error extracting tables from page {page_num+1}: {e}")
                    logging.error(f"Error extracting tables from page {page_num+1}: {e}")
                try:
                    # Extract images and run OCR
                    for img_idx, img_dict in enumerate(page.images):
                        try:
                            # Clamp coordinates to page bounds
                            x0 = max(img_dict["x0"], 0)
                            top = max(img_dict["top"], 0)
                            x1 = min(img_dict["x1"], float(page.width))
                            bottom = min(img_dict["bottom"], float(page.height))
                            if x1 <= x0 or bottom <= top:
                                raise ValueError(f"Invalid clamped bounding box: {(x0, top, x1, bottom)}")
                            cropped = page.crop((x0, top, x1, bottom)).to_image(resolution=300)
                            pil_img = cropped.original
                            ocr_text = pytesseract.image_to_string(pil_img)
                            if ocr_text.strip():
                                content.append(f"[Image {page_num+1}-{img_idx+1} OCR]:\n" + ocr_text.strip())
                        except Exception as ocr_e:
                            st.error(f"❌ Error extracting/ocr image {img_idx+1} on page {page_num+1}: {ocr_e}")
                            logging.error(f"Error extracting/ocr image {img_idx+1} on page {page_num+1}: {ocr_e}")
                except Exception as e:
                    st.error(f"❌ Error processing images on page {page_num+1}: {e}")
                    logging.error(f"Error processing images on page {page_num+1}: {e}")
    except Exception as e:
        st.error(f"❌ Error extracting text/tables/images: {e}")
        logging.error(f"Error extracting text/tables/images from file {pdf_path}: {e}")
    return "\n\n".join(content) 