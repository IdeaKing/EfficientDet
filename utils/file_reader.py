# File readers
# Thomas Chia 12/6/2021


def read_files(file_name):
    """Reads each file line by line."""
    file_contents = []
    file = open(file_name)
    lines = file.readlines()
    for line in lines:
        file_contents.append(line.strip())
    return file_contents


def parse_label_file(path_to_label_file):
    """Parses file with labels and converts into dict. For object detection."""
    labels = open(path_to_label_file)
    label_dict = {}
    index = 0
    for label in labels:
        label_dict[str(label.strip())] = index
        index = index + 1
    return label_dict
