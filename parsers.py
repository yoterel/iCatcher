import xml.etree.ElementTree as ET
import logging


class BaseParser:
    def __init__(self):
        self.classes = {'away': 0, 'left': 1, 'right': 2, 'off': 3}

    def parse(self, file):
        """
        returns a list of lists. each list contains the frame number, valid_flag, class
        where:
        frame number is zero indexed
        valid_flag is 1 if this frame has valid annotation, and 0 otherwise
        class is either away, left, right or off.
        :param file: the label file to parse.
        :return:
        """
        raise NotImplementedError


class XmlParser(BaseParser):
    def __init__(self, ext, labels_folder):
        super.__init__()
        self.ext = ext
        self.labels_folder = labels_folder

    def parse(self, file):
        query = self.labels_folder.glob(file + self.ext)
        try:
            next_item = next(query)
        except StopIteration:
            logging.info("The file: " + str(file) + " was skipped, since no matching xml was found.")
            return -1
        return self.xml_parse(next_item, 30, True)

    def xml_parse(self, input_file, fps, encode=False):
        tree = ET.parse(input_file)
        root = tree.getroot()
        counter = 0
        frames = {}
        current_frame = ""
        for child in root.iter('*'):
            if child.text is not None:
                if 'Response ' in child.text:
                    current_frame = child.text
                    frames[current_frame] = []
                    counter = 16
                else:
                    if counter > 0:
                        counter -= 1
                        frames[current_frame].append(child.text)
            else:
                if counter > 0:
                    if child.tag == 'true':
                        frames[current_frame].append(1)  # append 1 for true
                    else:
                        frames[current_frame].append(0)  # append 0 for false
        responses = []
        for key, val in frames.items():
            [num] = [int(s) for s in key.split() if s.isdigit()]
            responses.append([num, val])
        sorted_responses = sorted(responses)
        if encode:
            encoded_responses = []
            for response in sorted_responses:
                frame_number = int(response[1][4]) + int(response[1][10]) * fps + int(response[1][8]) * 60 * fps + int(
                    response[1][6]) * 60 * 60 * fps
                encoded_responses.append([frame_number, response[1][14], response[1][16]])
            sorted_responses = encoded_responses
        # replace offs with aways, they are equivalent
        for i, item in enumerate(sorted_responses):
            if item[2] == 'off':
                item[2] = 'away'
                sorted_responses[i] = item
        return sorted_responses
