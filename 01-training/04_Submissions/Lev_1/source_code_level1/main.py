import sys
from sandbox_agent import main

if __name__ == "__main__":
    if "--level" not in sys.argv:
        sys.argv.extend(["--level", "1"])
    main()
