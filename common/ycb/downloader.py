import argparse
import requests
import bs4
import os


def main(output_path: str, url: str, column_to_grab: str):
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

    sub_path = os.path.join(output_path, column_to_grab)
    os.makedirs(sub_path, exist_ok=True)

    for row in table.find_all("tr"):
        cells = row.find_all("td")
        if len(cells) == 0:
            continue
        object_name = cells[0].contents[0].strip()
        link = cells[column_idx].find("a")
        if link is None:
            print(f'No Link found for: {object_name}')
            continue
        download_url = requests.compat.urljoin(url, link['href'])

        print(object_name, download_url)

        file_name = os.path.basename(link['href'])
        with open(os.path.join(sub_path, file_name), 'wb') as file_out:
            result = requests.get(download_url)
            file_out.write(result.content)


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

    args = parser.parse_args()

    main(args.output, args.url, args.column)
