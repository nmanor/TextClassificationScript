import datetime


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class GlobalParameters(metaclass=Singleton):
    def __init__(self):
        self.FILE_NAME = ""
        self.FEATURES = []
        self.NORMALIZATION = ""
        self.OUTPUT_DIR = ""
        self.METHODS = ""
        self.TRAIN_DIR = ""
        self.TEST_DIR = ""
        self.NORM_PATH = ""
        self.RESULTS_PATH = ""
        self.TRAIN_DATA = ["1"]
        self.LABELS = []
        self.MEASURE = []
        self.STYLISTIC_FEATURES = []
        self.SELECTION = []
        self.WORDCLOUD = False
        self.LANGUAGE = None
        self.PRINT_SELECTION = False
        self.IDF = []


def print_message(msg, num_tabs=0):
    if num_tabs > 0:
        print("\t" * num_tabs, end="")
    print("{} >> {}".format(datetime.datetime.now(), msg))


if __name__ == "__main__":
    pass
