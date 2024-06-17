class Dataset():
    def __init__(self, name=""):
        self.name = name

    @staticmethod
    def load_feature(case, feature_data):
        for frame_id, frame_data in enumerate(feature_data):
            for vehicle_id, vehicle_data in frame_data.items():
                case[frame_id][vehicle_id].update(vehicle_data)
        
        return case