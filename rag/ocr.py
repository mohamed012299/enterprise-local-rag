import pytesseract

from pdf2image import convert_from_path

# =========================================================
# OCR CONFIG
# =========================================================

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

POPPLER_PATH = (
    r"C:\Users\moham\Downloads\Release-26.02.0-0\poppler-26.02.0\Library\bin"
)

# =========================================================
# OCR FUNCTION
# =========================================================

def extract_text_with_ocr(pdf_path):

    text = ""

    try:

        images = convert_from_path(
            pdf_path,
            poppler_path=POPPLER_PATH
        )

        for image in images:

            extracted = pytesseract.image_to_string(
                image,
                lang="eng+ara"
            )

            text += extracted + "\n"

    except Exception as e:

        print("OCR Error:", e)

    return text