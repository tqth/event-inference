def one_hot_to_label(one_hot, label_names, threshold=1):
    label = []
    for i in range(len(one_hot)):
        if one_hot[i] >= threshold:
            if i < len(label_names):
                label.append(label_names[i])
    return label

def map_top_indices_to_filenames(image_filenames, top_indices):
    """
    Chuẩn hóa danh sách tên ảnh và ánh xạ từ top_indices sang filename.

    Args:
        image_filenames (List[Union[str, Tuple[str]]]): Danh sách tên ảnh có thể bị gói trong tuple.
        top_indices (np.ndarray): Mảng chỉ số ảnh quan trọng, shape (batch_size, n)

    Returns:
        List[List[str]]: Danh sách các tên ảnh quan trọng, từng album một.
    """
    # Làm phẳng nếu phần tử là tuple
    clean_filenames = [
        fname[0] if isinstance(fname, tuple) else fname
        for fname in image_filenames
    ]

    # Lấy top filenames từ chỉ số
    top_filenames = [
        [clean_filenames[idx] for idx in album_top]
        for album_top in top_indices
    ]
    
    return top_filenames
