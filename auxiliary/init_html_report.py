import sys

sys.path.append('/home/thibault/ssd/netvision/')
from HtmlGenerator import HtmlGenerator


def main():
    webpage = HtmlGenerator(path="master.html")

    for dataset in ["Shapenet"]:
        table = webpage.add_table(dataset)
        table.add_column("Num Primitives")
        table.add_column("Decoder")
        table.add_column("Reconstruction")
        table.add_column("Dirname")

    webpage.return_html(save_editable_version=True)


if __name__ == "__main__":
    main()