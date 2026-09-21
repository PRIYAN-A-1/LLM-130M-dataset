import sys
from llm.cli import main
if __name__ == "__main__":
    sys.argv.insert(1,"prepare-data")
    main()
