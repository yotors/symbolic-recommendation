"""Run the live recommendation laboratory with ``python -m recommendation``."""

import os


if __name__ == "__main__":
    os.environ["RECOMMENDATION_DISABLE_DEFAULT_LAB"] = "1"
    from .app.server import main

    main()
