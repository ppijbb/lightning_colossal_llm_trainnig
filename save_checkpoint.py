import os
from huggingface_hub import HfApi


api=HfApi()

api.upload_folder(
    repo_id=os.getenv("MODEL_SAVE_REPO"),
    folder_path="/data/llm_checkpoint/last.ckpt/",
    repo_type="model",
    multi_commits=True,
    token=os.getenv("HUGGINGFACE_AUTO_TOKEN"))
