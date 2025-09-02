
import subprocess
import signal
import os
import time
import socket
import requests
import shutil
import ollama


def find_free_port():
    """Find an unused port by binding to port 0."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def wait_for_server(base_url, timeout=30):
    """Wait until Ollama server is ready on given port."""
    url = f"{base_url}/api/tags"
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(url, timeout=1)
            if r.status_code == 200:
                return True
        except Exception:
            time.sleep(0.5)
    raise TimeoutError(f"Ollama server at {base_url} did not become ready in {timeout}s")


class Ollama:
    def __init__(self, model_tag: str):
        self.proc = None
        self.port = None
        self.client = None
        self.model_tag = model_tag

    def __enter__(self):
        self.port = find_free_port()

        # Start ollama bound to that port
        ollama_path = shutil.which('ollama')
        if ollama_path is None:
            raise RuntimeError("Could not find ollama executable")
        self.proc = subprocess.Popen(
            [ollama_path, "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            preexec_fn=os.setsid,  # so we can kill the whole process group
            env={
                "OLLAMA_HOST": self.base_url(),
                "HOME": os.environ["HOME"]}
        )
        wait_for_server(self.base_url())

        # Create a client
        self.client = ollama.Client(host=self.base_url())
        self.client.pull(self.model_tag)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.client:
            self.client._client.close()
            self.client = None
        if self.proc:
            self.proc.terminate()
            self.proc.wait(timeout=5)
            self.proc = None
        self.port = None

    def __call__(self, prompt: str):
        output = self.client.generate(self.model_tag, prompt=prompt)
        return output.response

    def base_url(self):
        return f"http://127.0.0.1:{self.port}"
