import argparse
import requests
import os
from bs4 import BeautifulSoup

BASE_URL="https://scrimmage.pokerbots.org/"

def load_session_value(path):
    with open(path, 'r') as file_in:
        return file_in.read().strip()

def make_request(path, session_value):
    headers = {"cookie": f"session={session_value}"}

    r = requests.get(BASE_URL + path, headers=headers)

    if r.status_code == 200:
        return BeautifulSoup(r.text)
    else:
        print(f'Request to {path} failed!')
        return None

def pull_rows(page):
    result = page.find_all('tr')
    print(result)

def main(session_path):
    # Load session value
    session_value = load_session_value(session_path)
    response = make_request('team/games', session_value)
    pull_rows(response)


if __name__ == "__main__":
    DEFAULT_SESSION_PATH = os.path.expanduser("~/.pokerbots_session")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--session_path",
        default=DEFAULT_SESSION_PATH,
        help=f"Path to session value. default: {DEFAULT_SESSION_PATH}",
    )

    args = parser.parse_args()

    main(args.session_path)
