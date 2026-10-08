import os
import time
import base64
from google import genai
from google.genai.errors import ClientError
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def caption_image(image_bytes: bytes, ext: str, max_retries: int = 3) -> str:
    for attempt in range(max_retries):
        try:
            interaction = client.interactions.create(
                model="gemini-3.8-flash",
                input=[
                    {"type": "text", "text": (
                        "Describe this figure/chart from a research paper precisely, "
                        "including any visible labels, trends, or data values. 2-3 sentences."
                    )},
                    {"type": "image", "data": base64.b64encode(image_bytes).decode("utf-8"),
                     "mime_type": f"image/{ext}"},
                ],
            )
            return interaction.output_text
        except ClientError:
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                print(f"  Server busy, retrying in {wait}s...")
                time.sleep(wait)
            else:
                return "[Caption unavailable — service error]"