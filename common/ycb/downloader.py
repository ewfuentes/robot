import argparse
import requests
import bs4
import os
import tarfile


def download_ycb(output_path: str, url: str, column_to_grab: str, force: bool, verbose=False):
    result: requests.Response = requests.get(url)
    assert result.status_code == 200

    html = bs4.BeautifulSoup(result.text, "html.parser")
    object_grid = html.find(id="object_grid")
    table = object_grid.find("table")
    assert table is not None
    headers = table.find_all("th")
    column_idx = [i for i, x in enumerate(headers) if x.string == column_to_grab]
    assert (
        len(column_idx) > 0
    ), f'Unable to find "{column_to_grab}" in {[x.string for x in headers]}'
    column_idx = column_idx[0]

    os.makedirs(output_path, exist_ok=True)

    for row in table.find_all("tr"):
        cells = row.find_all("td")
        if len(cells) == 0:
            continue
        object_name = cells[0].contents[0].strip()
        link = cells[column_idx].find("a")
        if link is None:
            if verbose:
                print(f"No Link found for: {object_name}")
            continue
        download_url = requests.compat.urljoin(url, link["href"])

        file_name = os.path.basename(link["href"])
        file_path = os.path.join(output_path, file_name)
        if os.path.exists(file_path) and not force:
            if verbose:
                print(f'{object_name} already exists at {file_path}. Skipping...')
            continue

        if verbose:
            print(f'Downloading {object_name} to {file_path}...')

        with open(file_path, "wb") as file_out:
            result = requests.get(download_url)
            file_out.write(result.content)

        tgz = tarfile.open(file_path)
        tgz.extractall(output_path)
        tgz.close()

    # Delete the mtl file and rename the texture
    for path, subdirs, files in os.walk(output_path):
        if 'texture_map.png' in files:
            os.rename(os.path.join(path, 'texture_map.png'), os.path.join(path, 'textured.png'))
            with open(os.path.join(path, 'textured.mtl'), 'w') as file_out:
                file_out.write('newmtl material_0\n# shader_type beckmann\nmap_Kd textured.png')


if __name__ == "__main__":
    DEFAULT_URL = "http://ycb-benchmarks.s3-website-us-east-1.amazonaws.com/"
    DEFAULT_COLUMN = "16k laser scan"
    parser = argparse.ArgumentParser("Script to download YCB dataset")

    parser.add_argument(
        "--output", help="path to folder where objects should be saved", required=True
    )
    parser.add_argument(
        "--url", help=f"URL to object list. default: {DEFAULT_URL}", default=DEFAULT_URL
    )
    parser.add_argument(
        "--column",
        help=f"Column to grab. default: {DEFAULT_COLUMN}",
        default=DEFAULT_COLUMN,
    )
    parser.add_argument("--force", help="Force download if already exists.", action='store_true')
    parser.add_argument("--verbose", action='store_true')

    args = parser.parse_args()

    download_ycb(args.output, args.url, args.column, args.force, args.verbose)
