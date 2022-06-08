class BaseSplitter:
    def __init__(self, **kwargs):
        self.params = kwargs

    def split_two(self, df, train_frac):
        assert 0 < train_frac < 1

    def split_three(self, df, train_frac, val_frac):
        assert 0 < val_frac < train_frac < 1
        assert val_frac + train_frac < 1