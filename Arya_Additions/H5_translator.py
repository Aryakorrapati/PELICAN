import h5py

file_path = "C:/Users/aryak/OneDrive/Desktop/PELICAN/data/sample_data/run12/test.h5"
with h5py.File(file_path, "r") as f:
    def print_structure(name, obj):
        print(name) 

    f.visititems(print_structure)  

with h5py.File(file_path, "r") as f:
    for key in f.keys(): 
        print(f"Dataset: {key}")
        data = f[key][()] 
        print(data)