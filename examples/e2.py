from pyagentprolog_hf import PyAgentPrologConfig, PyAgentPrologModel

config = PyAgentPrologConfig(
    program="""
        родитель(иван, алексей).
        родитель(иван, мария).
        родитель(алексей, петр).
        родитель(алексей, анна).

        дедушка(Д, В) :- родитель(Д, Р), родитель(Р, В).
    """,
)

model = PyAgentPrologModel(config)

out = model.generate(["дедушка(иван, X)."])
print(out.solutions)
# → [[{'X': 'петр'}, {'X': 'анна'}]]

print(out.count)
# → 2