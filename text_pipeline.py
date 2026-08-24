"""Import the bundled textbooks into the local Knowledge Garden index."""

from core.retrieval import ingest_pdf_directory
from core.storage import GardenStore


if __name__ == "__main__":
    print(ingest_pdf_directory("./data/textbook_kb", GardenStore()))
