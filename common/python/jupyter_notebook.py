import re
import sys
from pathlib import Path

from jupyterlab import labapp

if __name__ == '__main__':
    print(labapp.__file__)
    app_dir = Path(labapp.__file__).parent / '..' / '..' / 'data/share/jupyter/lab'
    app_dir = app_dir.resolve()
    sys.argv[0] = re.sub(r'(-script\.pyw?|\.exe)?$', '', sys.argv[0])
    sys.argv.extend(['--app-dir', str(app_dir)])
    print(sys.argv)
    sys.exit(labapp.main())
