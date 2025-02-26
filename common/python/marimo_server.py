
import sys
from pathlib import Path
from marimo._cli.cli import main

if __name__ == "__main__":
    curr_path = Path(sys.argv[0])
    for p in curr_path.parent.iterdir():
        print(p)
    print(curr_path.parent)
    print('current path:', Path.cwd())
    print(sys.argv)
    sys.exit(main())
