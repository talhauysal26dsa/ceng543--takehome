from src.data import load_hotpotqa

documents, examples = load_hotpotqa(sample_size=5, split="validation", seed=42)

print(f"Loaded {len(documents)} documents and {len(examples)} examples\n")

for i, example in enumerate(examples[:3]):
    print(f"Example {i+1}:")
    print(f"  Question: {example.question}")
    print(f"  Answer: {example.answer}")
    print(f"  Relevant titles: {example.relevant_titles}")
    print(f"  Relevant titles type: {type(example.relevant_titles)}")
    print(f"  Relevant titles list: {list(example.relevant_titles)}")
    print()

print("\nFirst 5 documents:")
for doc in documents[:5]:
    print(f"  {doc.doc_id}: {doc.title}")
