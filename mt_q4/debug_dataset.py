from datasets import load_dataset

ds = load_dataset("hotpot_qa", "distractor", split="validation")
print(f"Dataset size: {len(ds)}")
print(f"\nFirst example keys: {ds[0].keys()}")
print(f"\nFirst example:")
example = ds[0]
for key, value in example.items():
    print(f"\n{key}:")
    if isinstance(value, (list, dict)):
        if key == "supporting_facts":
            print(f"  Type: {type(value)}")
            print(f"  Content: {value}")
        else:
            print(f"  Type: {type(value)}")
            if isinstance(value, dict):
                print(f"  Keys: {value.keys()}")
                for k, v in value.items():
                    if isinstance(v, list) and len(v) > 0:
                        print(f"    {k}: (length {len(v)}) First item: {v[0]}")
    else:
        print(f"  {value}")
