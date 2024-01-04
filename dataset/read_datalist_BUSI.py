import os


def read_datalist(path):
    samples_path = []
    labels = []
    for filename in os.listdir(path):
        for image in os.listdir(path+'/'+filename):
            labels.append(filename)
            samples_path.append(path+'/'+filename+'/'+image)
    replace_dict = {'normal': 0, 'benign': 1, 'malignant': 2}
    labels = [replace_dict.get(w, w) for w in labels]
    return samples_path, labels
