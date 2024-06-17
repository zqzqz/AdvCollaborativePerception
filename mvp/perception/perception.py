class Perception():
    def __init__(self):
        pass

    def run(self, case, **kwargs):
        raise NotImplementedError()

    def run_single_vehicle(self, case, **kwargs):
        raise NotImplementedError()

    def run_multi_vehicle(self, case, **kwargs):
        raise NotImplementedError()

    def run_multi_frame(self, case, **kwargs):
        raise NotImplementedError()