import torch
import numpy as np
import gc
from torchvision import transforms
import clip
from tqdm import tqdm

clip_model, clip_preprocess = clip.load("ViT-B/32", device='cuda')
clip_model.eval()

class CLIPRAMFeatureInferencer:
    def __init__(self, ram_model, clip_model, clip_preprocess, inference_ram, device='cuda', embedding_dim=512):
        """
        Dùng cho inference: chỉ trích xuất đặc trưng thị giác và đặc trưng văn bản từ RAM + CLIP.

        Args:
            ram_model: Mô hình RAM đã huấn luyện.
            clip_model: Mô hình CLIP với encode_image() và encode_text().
            clip_preprocess: Hàm tiền xử lý ảnh cho CLIP.
            inference_ram: Hàm sinh tag từ mô hình RAM.
            device: 'cuda' hoặc 'cpu'.
            embedding_dim: Kích thước embedding (mặc định 512 với CLIP ViT-B/32).
        """
        self.ram_model = ram_model
        self.clip_model = clip_model
        self.clip_preprocess = clip_preprocess
        self.inference_ram = inference_ram
        self.device = device
        self.embedding_dim = embedding_dim
        self.to_pil = transforms.ToPILImage()

        self.visual_embeddings = []
        self.text_embeddings = []
        self.combine_embeddings = []

    def get_all_tags(self, tag_importance_list):
        """
        Trích xuất tất cả tags từ RAM output.

        Args:
            tag_importance_list: List chứa dict {tag: importance score}

        Returns:
            List[str]: danh sách các tags.
        """
        tag_importance = tag_importance_list[0]
        return list(tag_importance.keys())

    def process(self, dataloader):
        """
        Trích xuất đặc trưng ảnh (CLIP) + tags (RAM) từ album ảnh để đưa vào mô hình phân loại.

        Args:
            dataloader: DataLoader chứa ảnh của album inference.
        """
        self.visual_embeddings = []
        self.text_embeddings = []
        self.combine_embeddings = []

        for album_images in tqdm(dataloader, desc="Extracting features for inference"):
            album_images = album_images.squeeze(0).to(self.device)

            # Tiền xử lý ảnh cho CLIP
            processed_images = torch.stack([
                self.clip_preprocess(self.to_pil(img)) for img in album_images
            ]).to(self.device)

            # Đặc trưng ảnh (CLIP image encoder)
            with torch.no_grad():
                visual_embeds = self.clip_model.encode_image(processed_images).cpu().numpy()

            self.visual_embeddings.append(visual_embeds)

            # Trích xuất tag từ RAM + mã hóa text bằng CLIP
            text_embeds = []
            for i in range(album_images.shape[0]):
                single_tensor = album_images[i:i+1]
                with torch.no_grad():
                    _, _, tag_importance = self.inference_ram(single_tensor, self.ram_model)

                all_tags = self.get_all_tags(tag_importance)
                joined_text = " ".join(all_tags)

                tokenized = clip.tokenize([joined_text], truncate=True).to(self.device)
                with torch.no_grad():
                    text_feat = self.clip_model.encode_text(tokenized).cpu().numpy()[0]

                text_embeds.append(text_feat)

                del single_tensor
                torch.cuda.empty_cache()

            text_embeds = np.stack(text_embeds)
            self.text_embeddings.append(text_embeds)

            # Gộp đặc trưng
            combined = np.concatenate([visual_embeds, text_embeds], axis=1)
            self.combine_embeddings.append(combined)

            del album_images, processed_images
            gc.collect()

        # Gộp toàn bộ embedding lại thành numpy arrays
        self.visual_embeddings = np.concatenate(self.visual_embeddings, axis=0)
        self.text_embeddings = np.concatenate(self.text_embeddings, axis=0)
        self.combine_embeddings = np.concatenate(self.combine_embeddings, axis=0)
        print("Feature extracted successfully!")
