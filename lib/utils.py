import time

def test_a_python_import_from_github():
  print('Success!')


class StepTimer(object):
    def __init__(self, description):
        self.description = description
    def __enter__(self):
        self.start = time.time()
    def __exit__(self, type, value, traceback):
        self.end = time.time()
        print(f"{self.description}: {self.end - self.start}")
