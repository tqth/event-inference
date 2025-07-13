import os
import urllib.request

def download_ram_weights(save_path="weights/ram_swin_large_14m.pth"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    if os.path.exists(save_path):
        print(f"✅ RAM weights already exist at: {save_path}")
        return save_path

    print("⏬ Downloading RAM weights...")
    url = "https://huggingface.co/spaces/xinyu1205/recognize-anything/resolve/main/ram_swin_large_14m.pth"
    try:
        urllib.request.urlretrieve(url, save_path)
        print(f"✅ Downloaded RAM weights to {save_path}")
    except Exception as e:
        print("❌ Failed to download RAM weights:", e)
        raise

    return save_path
