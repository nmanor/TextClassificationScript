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
        self.METHODS = ""
        self.DATASET_DIR = ""
        self.NORM_PATH = ""
        self.RESULTS_PATH = ""
        self.DATASET_DATA = ["1"]
        self.LABELS = []
        self.MEASURE = []
        self.STYLISTIC_FEATURES = []
        self.SELECTION = []
        self.WORDCLOUD = False
        self.LANGUAGE = None
        self.PRINT_SELECTION = False
        self.IDF = []
        self.K_FOLDS = 3
        self.ITERATIONS = 1
        self.BASELINE_PATH = ""
        self.EXPORT_AS_BASELINE = False


def print_message(msg, num_tabs=0):
    if num_tabs > 0:
        print("\t" * num_tabs, end="")
    print("{} >> {}".format(datetime.datetime.now(), msg))


if __name__ == "__main__":
    pass
