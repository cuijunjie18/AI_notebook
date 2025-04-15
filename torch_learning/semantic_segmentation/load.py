from my_frame import*

train_features, train_labels = read_voc_images(voc_dir, True)
print(train_features[0].shape)