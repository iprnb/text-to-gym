"""
Entry point – run this file to launch the Gradio app.

    python main.py
"""

from config import MODEL_PROVIDER, OLLAMA_API_KEY, OLLAMA_MODEL, OPENAI_API_KEY
from ui import demo


def _print_startup_info() -> None:
    width = 60
    print(f"\n{'='*width}")
    print("🚀 Starting RL Environment Design Pipeline")
    print(f"{'='*width}")
    print(f"Default Provider : {MODEL_PROVIDER}")
    print(f"Ollama API Key   : {'Set ✅' if OLLAMA_API_KEY else 'Not set ❌'}")
    print(f"OpenAI API Key   : {'Set ✅' if OPENAI_API_KEY else 'Not set ❌'}")
    print(f"Ollama Model     : {OLLAMA_MODEL}")
    print(f"{'='*width}\n")


if __name__ == "__main__":
    _print_startup_info()
    demo.launch(share=False, server_name="127.0.0.1", server_port=7860)
