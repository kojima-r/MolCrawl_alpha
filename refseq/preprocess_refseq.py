import os


def preprocess_refseq(root_folder):
    gz_files = find_gz_files(root_folder)
    print(gz_files)


def find_gz_files(directory):
    gz_files = []
    # Walk through the directory and subdirectories
    for root, _, files in os.walk(directory):
        for file_ in files:
            print(os.path.join(root, file_))
            if file_.endswith(".gz"):
                gz_files.append(os.path.join(root, file_))
    return gz_files

if __name__ == "__main__":

    refseq_root = "/nasa/datasets/riken/projects/fundamental_models_202407/refseq"
    preprocess_refseq(refseq_root)
