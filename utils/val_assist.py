from copy import deepcopy

import numpy as np
import pandas as pd
from tqdm import tqdm


class CustomStratifiedKFold:
    """
    from https://www.kaggle.com/frednavruzov/faster-stratified-cross-validation-v2

    Faster (yet memory-heavier) stratified cross-validation split
    Best suited for longer time-series with many different `y` groups
    """

    def __init__(
            self,
            n_splits: int = 5,
            shuffle: bool = True,
            random_state: int = 42
    ):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.seed = random_state
        self.folds_ = [(list(), list()) for _ in range(n_splits)]
        self.randomizer_ = np.random.RandomState(random_state)
        self.groups_ = None
        self.counts_ = None
        self.s_ = None

    def split(self, x, y):
        sorted_y = pd.Series(y).reset_index(drop=True).sort_values().astype('category').cat.codes
        self.s_ = pd.Series(data=sorted_y.index.values, index=sorted_y)
        self.groups_ = self.s_.index.unique()
        self.counts_ = np.bincount(self.s_.index)

        if self.n_splits > self.counts_.min():
            raise ValueError(
                f'Cannot split {self.counts_.min()} elements in smallest group on {self.n_splits} folds'
            )

        shift = 0
        for cnt in tqdm(self.counts_, desc='processing unique strats'):
            # get array of initial data's indices
            arr = self.s_.iloc[shift:shift + cnt].values
            # shuffle data if needed
            if self.shuffle:
                self.randomizer_.shuffle(arr)
            folds = np.array_split(arr, self.n_splits)
            # extend outer folds by elements from micro-folds
            for i in range(self.n_splits):
                cp = deepcopy(folds)
                # extend val indices
                val_chunk = cp.pop(i).tolist()
                self.folds_[i][1].extend(val_chunk)
                # extend train indices
                if self.shuffle:
                    cp = self.randomizer_.permutation(cp)
                train_chunk = np.hstack(cp).tolist()
                self.folds_[i][0].extend(train_chunk)

            # shift to the next group
            shift += cnt
        assert shift == len(self.s_)

        for (t, v) in self.folds_:
            yield (
                np.array(self.randomizer_.permutation(t) if self.shuffle else t, dtype=np.int32),
                np.array(self.randomizer_.permutation(v) if self.shuffle else v, dtype=np.int32)
            )

def batch_predict(df, num_iters, models):
    set_size = len(df)
    iterations = num_iters
    batch_size = set_size // iterations
    assert set_size == iterations * batch_size
    out = []
    for i in tqdm(range(num_iters)):
        if type(models) is list:
            preds = [model.predict(df.iloc[i*batch_size :(i+1)*batch_size]) for model in models]
            out.extend(np.mean(preds, axis=0))
        else:
            preds = models.predict(df.iloc[i*batch_size :(i+1)*batch_size])
            out.extend(np.mean(preds, axis=0))
    assert len(out) == set_size
    return out
