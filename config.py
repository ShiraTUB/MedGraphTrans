import os

from dotenv import load_dotenv

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # Project Root

load_dotenv()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
