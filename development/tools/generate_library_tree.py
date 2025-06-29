import sys
import os
import shutil 

if __name__ == "__main__":

    lib_dir = sys.argv[1]

    deploment_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../deployment/")) 
    destination_dir = os.path.join(lib_dir, "deep_microcompression")

    if os.path.exists(lib_dir) and os.path.exists(deploment_dir):
        if os.path.exists(destination_dir): 
            print(f"deep_micropression lib aleady exists in {lib_dir}")
        else:
            shutil.copytree(deploment_dir, destination_dir)
            print(f"Create deep_microprocompression lib in {lib_dir}")
    else:
        raise NotADirectoryError(f"{lib_dir} doesnot exist")