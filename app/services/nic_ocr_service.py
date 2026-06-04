import re
from functools import lru_cache

import easyocr


@lru_cache(maxsize=1)
def get_ocr_reader():
    # Runs locally on CPU
    return easyocr.Reader(["en"], gpu=False)


def normalize_nic_number(value: str) -> str:
    """
    Supports:
    - New NIC: 12 digits, example: 197264400700
    - Old NIC: 9 digits + V/X, example: 921234567V
    """
    if not value:
        return ""

    normalized = value.upper()

    normalized = normalized.replace(" ", "")
    normalized = normalized.replace("-", "")
    normalized = normalized.replace(".", "")
    normalized = normalized.replace("/", "")
    normalized = normalized.replace(":", "")
    normalized = normalized.replace("_", "")

    # Common OCR mistakes
    normalized = normalized.replace("O", "0")
    normalized = normalized.replace("I", "1")
    normalized = normalized.replace("L", "1")
    normalized = normalized.replace("S", "5")
    normalized = normalized.replace("B", "8")

    # Keep only digits and V/X
    normalized = re.sub(r"[^0-9VX]", "", normalized)

    return normalized


def extract_possible_nic_numbers_from_text(text: str) -> list[str]:
    compact_text = normalize_nic_number(text)

    # Overlapping extraction is important.
    # Example:
    # OCR text: 7197264400700
    # Possible 12 digit values:
    # 719726440070
    # 197264400700
    new_nic_matches = re.findall(r"(?=(\d{12}))", compact_text)

    # Old NIC format: 9 digits + V/X
    old_nic_matches = re.findall(r"(?=(\d{9}[VX]))", compact_text)

    possible_matches = new_nic_matches + old_nic_matches

    unique_matches = []

    for match in possible_matches:
        normalized = normalize_nic_number(match)

        if normalized not in unique_matches:
            unique_matches.append(normalized)

    return unique_matches


def verify_nic_number_from_document(
    document_image_path: str,
    submitted_nic_number: str,
) -> dict:
    try:
        submitted_nic = normalize_nic_number(submitted_nic_number)

        reader = get_ocr_reader()

        ocr_results = reader.readtext(
            document_image_path,
            detail=0,
            paragraph=False,
        )

        extracted_text = " ".join(ocr_results)

        extracted_nic_numbers = extract_possible_nic_numbers_from_text(
            extracted_text
        )

        # Exact match required
        if submitted_nic in extracted_nic_numbers:
            return {
                "nicMatched": True,
                "extractedNicNumber": submitted_nic,
                "reason": "Submitted NIC number matches the document NIC number.",
            }

        # OCR found a NIC number, but it is different
        if extracted_nic_numbers:
            return {
                "nicMatched": False,
                "extractedNicNumber": extracted_nic_numbers[0],
                "reason": "Submitted NIC number does not match the document NIC number.",
            }

        # OCR could not find NIC number
        return {
            "nicMatched": False,
            "extractedNicNumber": None,
            "reason": "Could not extract NIC number from the document image.",
        }

    except Exception as error:
        return {
            "nicMatched": False,
            "extractedNicNumber": None,
            "reason": f"NIC OCR verification failed: {str(error)}",
        }