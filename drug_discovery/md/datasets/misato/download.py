import os

if not os.path.exists("/data/Misato"):
    os.makedirs("/data/Misato", exist_ok=True)

os.system("apt-get install aria2")

os.system(
    "aria2c -x 16 -s 16 -d /data/Misato https://zenodo.org/record/7711953/files/MD.hdf5"
)
