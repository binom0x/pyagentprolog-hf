# pyagentprolog-hf

Hugging Face integration for [PyAgentProlog](https://github.com/yourusername/pyagentprolog).

## Author

Andrey Vorfolomeev

Allows you to use the PyAgentProlog Prolog engine as a regular HF model:

* `PyAgentPrologModel.from_pretrained(...)` / `save_pretrained(...)`
* `AutoModel.from_pretrained(..., trust_remote_code=True)`
* `pipeline("prolog", model=...)`
* `push_to_hub(...)` for publishing models to the HF Hub

## Installation

```bash
# The core is installed automatically as a dependency
pip install pyagentprolog-hf

# With semantic unification (sentence-transformers)
pip install pyagentprolog-hf[semantic-st]

# With Ollama embeddings
pip install pyagentprolog-hf[semantic-ollama]

# With CLI for HF Hub
pip install pyagentprolog-hf[hub]

# Everything at once
pip install pyagentprolog-hf[all]
```

## License

MIT