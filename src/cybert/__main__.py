"""Allow ``python -m cybert`` as a shortcut for ``python -m cybert.cli``."""

from .cli import main

if __name__ == "__main__":
    main()
