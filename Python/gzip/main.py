import gzip
import shutil


def gzip_file(input_file, output_file=None):
    if output_file is None:
        output_file = f"{input_file}.gz"

    with open(input_file, 'rb') as f_in:
        with gzip.open(output_file, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

    print(f"Compressed {input_file} to {output_file}")


gzip_file(r"C:\Users\Marcus\go\src\awesomeProject\styles.css")
