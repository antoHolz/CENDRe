class HFDataset:
    def __init__(self, 
                 dataset, 
                 pretransform=None,
                 transform=None,
                 posttransform=None,
                 xcols=[],
                 ycols=[],
                 withId=False
                 ):
        """
        Inherits from Hugging Face Dataset and applies pretransforms
        on each example when accessed.

        Args:
            pretransform (list of callables, optional): A list of functions
                that each take an example (dict) and return a (possibly modified)
                example.
            transform (callable, optional): A function that takes an example
            split_ratio (list of floats, optional): A list of floats that sum to 1.0
                which determine the ratio of examples to be used for each split.
            split (int, optional): The index of the split to use.
            percentage (float, optional): The percentage of examples to use.
            dataset_ratio (float, optional): The ratio of the dataset to use.
            shuffle (bool, optional): Whether to shuffle the dataset.
            scale (dict or list, optional): to which columns or positions to apply the scaler
            root (str, optional): The root directory to save the scaler and the dataset.
            name (str, optional): The name of the dataset.

        """
        # Initialize the base Dataset
        self.xcols = xcols
        self.ycols = ycols
        all_cols = dataset.features
        for col in all_cols:
            if col not in xcols and col not in ycols:
                dataset = dataset.remove_columns(col)
        self.dataset = dataset
        self.len = len(dataset)
        # Save parameters
        self.pretransform = pretransform or []
        self.transform = transform
        self.posttransform = posttransform or []
        self.withId = withId
            
    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        # Retrieve the example using the parent class's __getitem__
        example = self.dataset[idx]

        # Apply each pretransform sequentially
        if (self.pretransform is not None) and (callable(self.pretransform)):
            example = self.pretransform(example)
        
        # Apply the transform
        if (self.transform is not None) and (callable(self.transform)):
            example = self.transform(example)
              
        # Apply any DS spacific postttransform (e.g. for forecasting split)
        if (self.posttransform is not None) and (callable(self.posttransform)):
            example = self.posttransform(example)

        # Add id information if required (e.g. for CE tasks in ECLAD)
        if self.withId:
            example['id'] = idx
        return example
