# predict_event.py

from models.attention_model import AttentionNetworkMultiLabel
from models.clip_ram_inferencer import CLIPRAMFeatureInferencer
from dataset.album_dataset import AlbumInferenceDataset
from utils.label_utils import one_hot_to_label, map_top_indices_to_filenames
import torch
from torch.utils.data import DataLoader
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'

label_names = ['ThemePark', 'UrbanTrip', 'BeachTrip', 'NatureTrip', 'Zoo', 'Cruise', 'Show', 'Sports', 'PersonalSports',
         'PersonalArtActivity', 'PersonalMusicActivity', 'ReligiousActivity', 'GroupActivity', 'CasualFamilyGather',
         'BusinessActivity', 'Architecture', 'Wedding', 'Birthday', 'Graduation', 'Museum', 'Christmas', 'Halloween', 'Protest']

def load_models(device=device):
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
        embedding_dim=1024
    )

    return attention_model, extractor


def predict_event_and_top_n_images_from_folder(album_path, label_names = label_names, n_key_images = 5, device=device):
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
    features = extractor.combine_embeddings

    X_input = np.expand_dims(features, axis=0)
    X_input = X_input.astype('float32')
    print("X_input shape", X_input.shape)
    print("X_input type", X_input.dtype)

    # Dự đoán
    y_pred = attention_model.predict(X_input)
    labels = one_hot_to_label(y_pred[0], label_names)

    #Dự đoán ảnh quan trọng
    top_indices, top_weights = attention_model.get_top_n_important_images(X_input, n=n_key_images)
    top_filenames = map_top_indices_to_filenames(extractor.filenames_per_album[0], top_indices)

    return labels, top_filenames
