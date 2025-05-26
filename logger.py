import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

logger = logging.getLogger(__name__)

def print_section_info(message):
    logging.info("\n" + "=" * 30 + "\n" + message + "\n" + "=" * 30)
