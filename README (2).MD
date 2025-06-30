# SmartSDLC – AI-Enhanced Software Development Lifecycle

SmartSDLC is a local AI-powered dashboard that automates key phases of the Software Development Lifecycle (SDLC) using a lightweight instruction-following LLM. The system provides intelligent support for:

- ✅ Requirement Analysis  
- ✅ Code Generation  
- ✅ Test Case Creation  
- ✅ Bug Fixing  
- ✅ Documentation Generation  

---

## 🧠 Model Used

We use IBM’s **Granite 3B Instruct** model in GGUF format (`granite-3b-2b-instruct-Q4_K_M.gguf`). This quantized model runs locally using `llama-cpp-python`, enabling efficient, GPU-free inference.

> **Note**: The model file is **not included** in the repository due to its size.

### 📥 Download the Model

1. Go to the official Hugging Face page:  
   [https://huggingface.co/ibm-granite/granite-3b-2b-instruct-GGUF](https://huggingface.co/ibm-granite/granite-3b-2b-instruct-GGUF)

2. Download the file:  
   `granite-3b-2b-instruct-Q4_K_M.gguf`

3. Place it in the following directory:  
