import gzip
import pickle
def load_model_from_gz(file_path):
    with gzip.open(file_path, 'rb') as f:
        model = pickle.load(f)
    return model