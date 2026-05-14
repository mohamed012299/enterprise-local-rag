import tempfile

from langchain_core.documents import Document

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader
)

from rag.ocr import extract_text_with_ocr

# =========================================================
# LOAD FILES
# =========================================================

def load_uploaded_files(uploaded_files, st):

    all_documents = []

    processed_files = []

    for uploaded_file in uploaded_files:

        file_extension = (
            uploaded_file.name
            .split(".")[-1]
            .lower()
        )

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=f".{file_extension}"
        ) as tmp_file:

            tmp_file.write(
                uploaded_file.read()
            )

            temp_path = tmp_file.name

        try:

            # =====================================================
            # PDF
            # =====================================================

            if file_extension == "pdf":

                loader = PyPDFLoader(
                    temp_path
                )

                documents = loader.load()

                # Check extracted text
                pdf_text = ""

                for doc in documents:

                    if doc.page_content:

                        pdf_text += (
                            doc.page_content.strip()
                        )

                # OCR fallback
                if len(pdf_text) < 20:

                    st.warning(
                        f"⚡ Using OCR for scanned PDF: {uploaded_file.name}"
                    )

                    ocr_text = (
                        extract_text_with_ocr(
                            temp_path
                        )
                    )

                    if ocr_text.strip():

                        documents = [
                            Document(
                                page_content=ocr_text
                            )
                        ]

                    else:

                        st.error(
                            f"❌ OCR failed for {uploaded_file.name}"
                        )

                        continue

            # =====================================================
            # TXT
            # =====================================================

            elif file_extension == "txt":

                loader = TextLoader(
                    temp_path,
                    encoding="utf-8"
                )

                documents = loader.load()

            # =====================================================
            # DOCX
            # =====================================================

            elif file_extension == "docx":

                loader = Docx2txtLoader(
                    temp_path
                )

                documents = loader.load()

            else:
                continue

            all_documents.extend(
                documents
            )

            processed_files.append(
                uploaded_file.name
            )

        except Exception as e:

            st.warning(
                f"Could not process {uploaded_file.name}"
            )

    return all_documents, processed_files