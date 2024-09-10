import ncbi_genome_download as ngd


def download(group, nb_try=0, max_try=5):
    try:
        ngd.download(
            groups=group,
            section="refseq",
            output=output_path,
            parallel=4,
            file_formats="fasta",
            progress_bar=True,
        )
    except ConnectionError as e:
        if nb_try < max_try:
            print(f"ConnectionError Retrying ({nb_try + 1})")
            return download(nb_try + 1)
        else:
            raise e


if __name__ == "__main__":

    output_path = "/nasa/datasets/riken/projects/fundamental_models_202407/"
    groups = [
        "archaea",
        "bacteria",
        "fungi",
        "invertebrate",
        "metagenomes",
        "plant",
        "protozoa",
        "vertebrate_mammalian",
        "vertebrate_other",
        "viral",
    ]
    for group in groups:
        download(group)

    # ngd.download(section="refseq", output=output_path, parallel=4, progress_bar=True, format="fasta")
