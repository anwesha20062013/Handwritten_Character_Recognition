def load_mapping(file_path="emnist-balanced-mapping.txt"):

    mapping = {}

    with open(file_path, "r") as f:

        for line in f:

            parts = line.strip().split()

            label = int(parts[0])

            ascii_code = int(parts[1])

            mapping[label] = chr(ascii_code)

    return mapping