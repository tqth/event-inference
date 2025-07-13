def one_hot_to_label(one_hot, label_names, threshold=1):
    label = []
    for i in range(len(one_hot)):
        if one_hot[i] >= threshold:
            if i < len(label_names):
                label.append(label_names[i])
    return label