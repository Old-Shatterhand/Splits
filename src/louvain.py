import pandas as pd
import networkx as nx
import numpy as np
import community
from src.base import BaseSplitter


class LouvainSplitter(BaseSplitter):
    def __init__(self, **kwargs):
        super(LouvainSplitter, self).__init__(**kwargs)

    @staticmethod
    def get_all():
        return [
            LouvainSplitter(),
        ]

    def split_two(self, df, train_frac):
        super(LouvainSplitter, self).split_two(df, train_frac)
        df = self._get_communities(df)
        df = self._split_groups(df, "community", 10, train_frac, 0)
        return df

    def split_three(self, df, train_frac, val_frac):
        super(LouvainSplitter, self).split_three(df, train_frac, val_frac)
        df = self._get_communities(df)
        df = self._split_groups(df, "community", 10, train_frac, 0)
        return df

    @staticmethod
    def _split_groups(
            inter: pd.DataFrame,
            col_name: str = "Target_ID",
            bin_size: int = 10,
            train_frac: float = 0.7,
            val_frac: float = 0.2,
    ) -> pd.DataFrame:
        """Split data by protein (cold-target)
        Tries to ensure good size of all sets by sorting the prots by number of interactions
        and performing splits within bins of 10
        Args:
            inter (pd.DataFrame): interaction DataFrame
            col_name (str): Which column to split on (col_name or 'Drug_ID' usually)
            bin_size (int, optional): Size of the bins to perform individual splits in. Defaults to 10.
            train_frac (float, optional): value from 0 to 1, how much of the data goes into train
            val_frac (float, optional): value from 0 to 1, how much of the data goes into validation
        Returns:
            pd.DataFrame: DataFrame with a new 'split' column
        """
        sorted_index = [x for x in inter[col_name].value_counts().index]
        train_prop = int(bin_size * train_frac)
        val_prop = int(bin_size * val_frac)
        train = []
        val = []
        test = []
        for i in range(0, len(sorted_index), bin_size):
            subset = sorted_index[i: i + bin_size]
            train_bin = list(np.random.choice(subset, min(len(subset), train_prop), replace=False))
            train += train_bin
            subset = [x for x in subset if x not in train_bin]
            val_bin = list(np.random.choice(subset, min(len(subset), val_prop), replace=False))
            val += val_bin
            subset = [x for x in subset if x not in val_bin]
            test += subset
        train_idx = inter[inter[col_name].isin(train)].index
        val_idx = inter[inter[col_name].isin(val)].index
        test_idx = inter[inter[col_name].isin(test)].index
        inter.loc[train_idx, "split"] = "train"
        inter.loc[val_idx, "split"] = "val"
        inter.loc[test_idx, "split"] = "test"
        return inter

    @staticmethod
    def _get_communities(df: pd.DataFrame) -> pd.DataFrame:
        """Assigns each interaction to a community, based on Louvain algorithm."""
        G = nx.from_pandas_edgelist(df, source="Target_ID", target="Drug_ID")
        all_drugs = set(df["Drug_ID"].unique())
        all_prots = set(df["Target_ID"].unique())
        communities = community.best_partition(G)
        s = [set() for _ in range(max(communities.values()) + 1)]
        for k, v in communities.items():
            s[v].add(k)
        communities = []
        for i in s:
            drugs = i.intersection(all_drugs)
            prots = i.intersection(all_prots)
            subset = df[df["Target_ID"].isin(prots) & df["Drug_ID"].isin(drugs)]
            communities.append(
                {
                    "protn": len(prots),
                    "drugn": len(drugs),
                    "edgen": len(subset),
                    "protids": prots,
                    "drugids": drugs,
                }
            )
        communities = pd.DataFrame(communities).sort_values("edgen").reset_index(drop=True)
        for name, row in communities.iterrows():
            idx = df["Target_ID"].isin(row["protids"]) & df["Drug_ID"].isin(row["drugids"])
            df.loc[idx, "community"] = "com" + str(int(name))
        df = df[df["community"].notna()]
        return df
