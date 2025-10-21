from huggingface_hub import list_repo_files

files = list_repo_files("ayamohamed2500/seqtrack-checkpoints", repo_type="model")
for f in files:
    print(f)
