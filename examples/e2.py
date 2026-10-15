from pyagentprolog_hf import PyAgentPrologConfig, PyAgentPrologModel

config = PyAgentPrologConfig(
    program="""
        parent(ivan, alexey).
        parent(ivan, maria).
        parent(alexey, peter).
        parent(alexey, anna).

        grandfather(G, C) :- parent(G, P), parent(P, C).
    """,
)

model = PyAgentPrologModel(config)

out = model.generate(["grandfather(ivan, X)."])
print(out.solutions)
# → [[{'X': 'peter'}, {'X': 'anna'}]]

print(out.count)
# → 2