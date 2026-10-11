from pyagentprolog_hf import PyAgentPrologModel

model = PyAgentPrologModel.from_pretrained(
    "your-username/pyagent-prolog-demo",
    trust_remote_code=True,
)