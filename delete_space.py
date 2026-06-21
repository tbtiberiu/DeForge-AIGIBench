import os

from dotenv import load_dotenv
from huggingface_hub import HfApi

# Try loading from .env
load_dotenv(override=True)

token = os.getenv('HF_TOKEN_WRITE') or os.getenv('HF_TOKEN')
repo_id = 'TheKernel01/DeForge-AIGIBench-Space'

if not token:
    print('Could not find HF_TOKEN_WRITE or HF_TOKEN in your .env file.')
    token = input('Please paste your Hugging Face WRITE token: ').strip()

if not token:
    print('Error: A token is required to delete/recreate the Space.')
    exit(1)

api = HfApi()

print(f"\n1. Attempting to delete space '{repo_id}'...")
try:
    api.delete_repo(repo_id=repo_id, repo_type='space', token=token)
    print('Success: Space deleted successfully!')
except Exception as e:
    print(
        f'Notice: Could not delete space (it might not exist yet, or there was a platform error): {e}'
    )

print(f"\n2. Attempting to recreate space '{repo_id}'...")
try:
    # api.create_repo(repo_id=repo_id, repo_type='space', space_sdk='gradio', token=token)
    print('Success: Space created successfully!')
    print('\nYou can now trigger your GitHub Actions workflow to push the code.')
except Exception as e:
    print(f'Error: Failed to create space: {e}')
