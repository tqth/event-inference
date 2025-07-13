# predict_event.py

from models.attention_model import AttentionNetworkMultiLabel
from models.clip_ram_inferencer import CLIPRAMFeatureInferencer
from dataset.album_dataset import AlbumInferenceDataset
from utils.label_utils import one_hot_to_label

from torch.utils.data import DataLoader
import numpy as np

label_names = ['ThemePark', 'UrbanTrip', 'BeachTrip', 'NatureTrip', 'Zoo', 'Cruise', 'Show', 'Sports', 'PersonalSports',
         'PersonalArtActivity', 'PersonalMusicActivity', 'ReligiousActivity', 'GroupActivity', 'CasualFamilyGather',
         'BusinessActivity', 'Architecture', 'Wedding', 'Birthday', 'Graduation', 'Museum', 'Christmas', 'Halloween', 'Protest']

def load_models(device='cuda'):
    attention_model = AttentionNetworkMultiLabel(input_shape=(20, 1024), output_shape=23)
    attention_model.load_weights("weights/attention_model.h5")

    import clip
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
    from ram.models import ram
    from ram import inference_ram

    from utils.download_weights import download_ram_weights

    RAM_WEIGHTS_PATH = download_ram_weights()
    model = ram(pretrained=RAM_WEIGHTS_PATH, image_size=384, vit='swin_l')
    model = model.to(device)
    model.eval()
    # Init feature extractor
    extractor = CLIPRAMFeatureInferencer(
        ram_model=model,
        clip_model=clip_model,
        clip_preprocess=clip_preprocess,
        inference_ram=inference_ram,
        device=device,
        embedding_dim=512
    )

    return attention_model, extractor


def predict_event_from_folder(album_path, label_names = label_names, device='cuda'):
    """
    Dự đoán sự kiện cho một thư mục ảnh (album).

    Args:
        album_path (str): Đường dẫn thư mục ảnh
        label_names (List[str]): Danh sách tên nhãn
        device (str): 'cuda' hoặc 'cpu'

    Returns:
        List[str]: Các nhãn được dự đoán
    """
    attention_model, extractor = load_models(device=device)

    dataset = AlbumInferenceDataset(album_path=album_path, album_size=20)
    dataloader = DataLoader(dataset, batch_size=1)

    # Trích xuất đặc trưng
    extractor.process(dataloader)
    features = extractor.combine_embeddings  # (20, 1024)

    X_input = np.expand_dims(features, axis=0)  # (1, 20, 1024)

    # Dự đoán
    y_pred = attention_model.predict(X_input)
    labels = one_hot_to_label(y_pred[0], label_names)

    return labels
