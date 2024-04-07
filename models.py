from huggingface_hub import hf_hub_download, HfApi

def ensure_model_is_latest(repo_id, filename):
    """
    Ensure the local model is the latest version from Hugging Face Hub.

    Parameters:
    - repo_id (str): Repository ID on Hugging Face, e.g., 'username/model-name'.
    - filename (str): The name of the file to check, e.g., 'pytorch_model.bin'.

    Returns:
    - filepath (str): The path to the local model file.
    """
    # Instantiate the HF API
    api = HfApi()

    # Get the model's repo info to check for the latest file version
    repo_info = api.model_info(repo_id)
    # Assuming the model file is at the root of the repo
    file_info = next((f for f in repo_info.siblings if f.rfilename == filename), None)

    if file_info:
        # Download or update the file as needed
        filepath = hf_hub_download(repo_id=repo_id, filename=filename, force_filename=filename)
        return filepath
    else:
        raise ValueError(f"File '{filename}' not found in repository '{repo_id}'.")

# Example usage
repo_id = "username/model-name"
filename = "model_file.gguf"  # Adjust based on your model file name
model_filepath = ensure_model_is_latest(repo_id, filename)

# Now, model_filepath points to the latest version of your model file locally
