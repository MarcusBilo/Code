# pip install scour

import os
from scour.scour import start, getInOut, parse_args


def scour_svg(inputfile, outputfile):
    options = parse_args()
    options.infilename = inputfile
    options.outfilename = outputfile
    (inputfile, outputfile) = getInOut(options)
    start(options, inputfile, outputfile)


def compress_svg_files_in_folder(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(".svg"):
            input_file_path = os.path.join(folder_path, filename)
            output_file_path = os.path.join(folder_path, f"compressed_{filename}")
            scour_svg(input_file_path, output_file_path)


def main():
    folder_path = "path/to/your/folder"
    compress_svg_files_in_folder(folder_path)


if __name__ == "__main__":
    main()
