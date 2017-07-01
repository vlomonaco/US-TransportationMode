from TMDataset import TMDataset

class Processing:
    dataset = TMDataset()

    def __init__(self):
        self.dataset.create_balanced_dataset(True)

if __name__ == "__main__":
    processing = Processing()