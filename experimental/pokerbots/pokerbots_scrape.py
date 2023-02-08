import argparse
import requests
import os
from bs4 import BeautifulSoup
import re
import functools
from collections import namedtuple
import pandas as pd
import datetime

BASE_URL = "https://scrimmage.pokerbots.org/"


def load_session_value(path=os.path.expanduser("~/.pokerbots_session")):
    with open(path, "r") as file_in:
        return file_in.read().strip()


def make_request(path, session_value):
    headers = {"cookie": f"session={session_value}"}

    r = requests.get(BASE_URL + path, headers=headers)

    if r.status_code == 200:
        return BeautifulSoup(r.text, features="html.parser")
    else:
        print(f"Request to {path} failed!")
        return None


def get_page_range(session_value):
    page = make_request("team/games", session_value)
    page_num_regex = r"/team/games\?page=(\d+)"
    min_page = 1000000
    max_page = 0
    for link in page.find_all("a"):
        m = re.search(page_num_regex, link["href"])
        if m:
            min_page = min(min_page, int(m.group(1)))
            max_page = max(max_page, int(m.group(1)))
    return (min_page, max_page)


def clean_up_column_name(name):
    return name.replace(" ", "_").replace(".", "").lower()


def get_rows_from_page(page):
    table = page.find("table")
    table_header = table.thead
    fields = [clean_up_column_name(elem.string) for elem in table_header.find_all("th")]
    fields.append("game_id")
    RowType = namedtuple("Row", fields)

    rows = []
    for row in table.tbody.find_all("tr"):
        elements = []
        for cell in row.find_all("td"):
            if cell.a:
                elements.append(cell.a["href"])
            else:
                cell_data = cell.string.strip()
                try:
                    elements.append(int(cell_data))
                except:
                    try:
                        elements.append(float(cell_data))
                    except:
                        try:
                            elements.append(datetime.datetime.fromisoformat(cell_data))
                        except:
                            elements.append(cell_data)
        player_log = elements[-1]
        m = re.match(r"/team/game/(\d+)/player_log", player_log)
        if m:
            elements.append(int(m.group(1)))
            rows.append(RowType(*elements))
    return rows


def pull_rows(page_limit, session_value):
    page_range = get_page_range(session_value)

    rows = []
    for page_num in range(page_range[0], min(page_range[1], page_limit) + 1):
        results_page = make_request(f"team/games?page={page_num}", session_value)
        rows += get_rows_from_page(results_page)

    return pd.DataFrame(rows)


def parse_cards(card_str):
    return card_str[1:-1].split(" ")


def parse_game_round(game_round):
    GameRound = namedtuple(
        "GameRound",
        [
            "round_number",
            "A_player",
            "B_player",
            "A_initial_bankroll",
            "B_initial_bankroll",
            "A_hand",
            "B_hand",
            "round_actions",
            "board_states",
            "A_delta",
            "B_delta",
        ],
    )
    round_actions = []
    a_hand = []
    b_hand = []
    board_states = []
    for line in game_round.split("\n"):
        words = line.split(" ")
        if words[0] == "Round":
            # m = re.match(r'Round #\(\d+\) \(A|B\)(\(\d+\)) \(A|B\)(\(\d+\))', line)
            m = re.match(r"Round #(\d+), (A|B) \((-?\d+)\), (A|B) \((-?\d+)\)", line)
            round_number = int(m.group(1))
            player_1 = m.group(2)
            player_1_bank_roll = int(m.group(3))
            player_2 = m.group(4)
            player_2_bank_roll = int(m.group(5))

            round_actions.append([])
            board_states.append([])

        elif words[0] in "AB":
            if words[1] == "dealt":
                cards = parse_cards(re.search(r"(\[.*\])", line).group(1))
                if words[0] == "A":
                    a_hand = cards
                else:
                    b_hand = cards
            elif words[1] == "posts":
                round_actions[-1].append((words[0], "posts", int(words[-1])))
            elif words[1] in ["bets", "raises"]:
                round_actions[-1].append((words[0], "raises", int(words[-1])))
            elif words[1] == "awarded":
                if words[0] == "A":
                    a_delta = int(words[-1])
                else:
                    b_delta = int(words[-1])
            else:
                round_actions[-1].append(tuple(words))

        elif words[0] in ["Flop", "Turn", "River", "Run"]:
            round_actions.append([])
            cards = parse_cards(re.search(r"(\[.*\])", line).group(1))
            board_states.append(cards)
        else:
            print("Unknown line:", line)
            print(game_round)
    return GameRound(
        round_number=round_number,
        A_player=0 if player_1 == "A" else 1,
        B_player=0 if player_1 == "B" else 1,
        A_initial_bankroll=player_1_bank_roll
        if player_1 == "A"
        else player_2_bank_roll,
        B_initial_bankroll=player_1_bank_roll
        if player_1 == "B"
        else player_2_bank_roll,
        A_hand=a_hand,
        B_hand=b_hand,
        round_actions=round_actions,
        board_states=board_states,
        A_delta=a_delta,
        B_delta=b_delta,
    )


def parse_game_log(row, game_log):
    Game = namedtuple("Game", ['id', "a_bot", "b_bot", "rounds"])
    rounds = []
    for game_round in game_log.split("\n\n"):
        if not game_round.startswith("Round"):
            continue
        rounds.append(parse_game_round(game_round))
    return Game(id = row['game_id'], a_bot=row.challenger, b_bot=row.opponent, rounds=rounds)


def get_game_log(row, session_value):
    path = row.game_log
    game_log = make_request(path, session_value).get_text()
    return parse_game_log(row, game_log)


def main(session_path, page_limit):
    # Load session value
    session_value = load_session_value(session_path)
    return pull_rows(page_limit, session_value)


if __name__ == "__main__":
    DEFAULT_SESSION_PATH = os.path.expanduser("~/.pokerbots_session")
    DEFAULT_PAGE_LIMIT = 5
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--session_path",
        default=DEFAULT_SESSION_PATH,
        help=f"Path to session value. default: {DEFAULT_SESSION_PATH}",
    )
    parser.add_argument(
        "--page_limit",
        default=DEFAULT_PAGE_LIMIT,
        type=int,
        help=f"Number of pages to parse. default: {DEFAULT_PAGE_LIMIT}",
    )

    args = parser.parse_args()

    main(args.session_path, args.page_limit)
