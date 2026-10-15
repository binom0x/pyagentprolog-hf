from pyagentprolog_hf import PyAgentPrologConfig, PyAgentPrologModel

config = PyAgentPrologConfig(
    program="""
        parent(ivan, alexey).
        parent(alexey, peter).
        grandfather(G, GS) :- parent(G, P), parent(P, GS).
    """,
)
model = PyAgentPrologModel(config)

out = model.generate(["grandfather(ivan, X)."])
print(out.solutions)   # [[{'X': 'peter'}]]