def create_model_directory(model, model_name=None):
    import os
    import pickle
    import gzip
    model_filename='files/models'
    os.makedirs(model_filename,exist_ok=True)

    model_path=os.path.join(model_filename,model_name)
    with gzip.open(model_path,'wb') as file:
        pickle.dump(model,file)
