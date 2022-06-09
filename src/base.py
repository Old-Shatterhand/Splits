import json


class BaseSplitter:
    def __init__(self, **kwargs):
        """Initialize the splitting algorithm and save kwargs as hyperparameter of the algorithm"""
        self.params = kwargs

    @property
    def author(self):
        """The authors name of the splitting implementation"""
        return "John Doe"

    @property
    def parameter(self):
        """Get the parameters of the algorithm"""
        return json.dumps(self.params)

    @staticmethod
    def get_all():
        """Return a list with all hyperparameter setting you want to test"""
        raise NotImplementedError()

    def split_two(self, df, train_frac):
        """Split the given dataframe into train and test parts"""
        assert 0 < train_frac < 1

    def split_three(self, df, train_frac, val_frac):
        """Split the given dataframe into train, val, and test parts"""
        assert 0 < val_frac < train_frac < 1
        assert val_frac + train_frac < 1