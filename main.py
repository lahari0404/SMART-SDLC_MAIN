from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
from llama_cpp import Llama

# Load the local GGUF model
MODEL_PATH = "./models/granite-3.3-2b-instruct-Q4_K_M.gguf"  # Change to your file name
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=2048,
    n_threads=8,
    n_gpu_layers=20,
    temperature=0.7
)

# FastAPI app setup
app = FastAPI()

# Allow frontend access from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Helper function to run LLaMA inference
def run_llm(prompt: str) -> str:
    output = llm(
        prompt=prompt,
        stop=["</s>"],
        max_tokens=1024
    )
    return output["choices"][0]["text"].strip()

# ---- API Endpoints ----

@app.post("/analyze")
def analyze(input: str = Form(...)):
    prompt = f"Analyze and summarize this software requirement:\n{input}"
    return {"response": run_llm(prompt)}

@app.post("/generate-code")
def generate_code(input: str = Form(...)):
    prompt = f"Generate clean and efficient code for this task:\n{input}"
    return {"response": run_llm(prompt)}

@app.post("/create-tests")
def create_tests(input: str = Form(...)):
    prompt = f"Create test cases for the following function or feature:\n{input}"
    return {"response": run_llm(prompt)}

@app.post("/fix-bugs")
def fix_bugs(input: str = Form(...)):
    prompt = f"Fix bugs in the following code:\n{input}"
    return {"response": run_llm(prompt)}

@app.post("/generate-docs")
def generate_docs(input: str = Form(...)):
    prompt = f"Write documentation for this code or module:\n{input}"
    return {"response": run_llm(prompt)}
