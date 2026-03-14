import json

FILE = "data/trusted_sources.json"


def load_sources():
    with open(FILE) as f:
        data = json.load(f)
        return data["trusted_sources"]


def is_trusted(source):

    sources = load_sources()

    source = source.lower()

    for domain in sources:
        if domain in source:
            return True

    return False


def add_source(source):

    with open(FILE) as f:
        data = json.load(f)

    sources = data["trusted_sources"]

    if source not in sources:
        sources.append(source)

    with open(FILE, "w") as f:
        json.dump({"trusted_sources": sources}, f, indent=4)