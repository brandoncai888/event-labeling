import numpy as np
import pandas as pd
import h5py
import os
import sys

def npy_to_parquet(folder, p_file, t_file, xy_file, parquet_file):
    # Use os.path.join to prevent string concatenation errors
    p = np.load(os.path.join(folder, p_file))
    t = np.load(os.path.join(folder, t_file))
    xy = np.load(os.path.join(folder, xy_file))

    if not (len(p) == len(t) == len(xy)):
        raise ValueError("Array lengths do not match!")

    df = pd.DataFrame({
        "t": t,
        "x": xy[:, 0],
        "y": xy[:, 1],
        "p": p
    })
    
    output_path = os.path.join(folder, parquet_file)
    df.to_parquet(output_path, index=False)
    print(f"Saved {len(df)} events to {output_path}")

def h5_to_parquet(folder, h5_file, parquet_file):
    input_path = os.path.join(folder, h5_file)
    
    with h5py.File(input_path, "r") as f:
        # Recursive search for the first dataset
        def find_dataset(group):
            for k in group.keys():
                item = group[k]
                if isinstance(item, h5py.Dataset):
                    return item[:]
                if isinstance(item, h5py.Group):
                    res = find_dataset(item)
                    if res is not None: return res
            return None

        data = find_dataset(f)
        if data is None:
            raise ValueError("No dataset found in HDF5 file")

    df = pd.DataFrame(data)
    
    # Ensure columns are named correctly if the H5 wasn't structured
    if all(x in df.columns for x in ["t", "x", "y", "p"]):
        df = df[["t", "x", "y", "p"]]
    else:
        # Fallback: assume columns are in order 0=t, 1=x, 2=y, 3=p
        df.columns = ["t", "x", "y", "p"]

    output_path = os.path.join(folder, parquet_file)
    df.to_parquet(output_path, index=False)
    print(f"Saved to {output_path}")

def to_parquet(mode, *args):
    try:
        if mode == "h5":
            # Expects: folder, h5_file, parquet_file
            h5_to_parquet(*args)
        elif mode == "npy":
            # Expects: folder, p_file, t_file, xy_file, parquet_file
            npy_to_parquet(*args)
        else:
            print(f"Unknown mode: {mode}")
    except TypeError as e:
        print(f"Error: Argument mismatch for mode '{mode}'. Check your input count.")
        print(e)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py [h5|npy] [args...]")
    else:
        # The asterisk (*) here is vital to unpack the list into arguments
        to_parquet(*sys.argv[1:])