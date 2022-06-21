from src.genetic import GeneticSplitter
from pathlib import Path
import numpy as np
import pandas as pd
import time

from src.louvain import LouvainSplitter


def generate_random_graph(nodes=100, edges=1000):
    """Generate a random graph with the given number of nodes and edges"""
    with open(f"data/random_{nodes}x{edges}.tsv", "w") as data:
        s = np.random.normal(loc=0.5)
        while 0.6 < s or s < 0.4:
            s = np.random.normal(loc=0.5)
        pd_split = int(s * nodes)
        drugs = [f"D{i:05d}" for i in range(pd_split)]
        prots = [f"T{i:05d}" for i in range(nodes - pd_split)]
        data.write("Drug_ID\tTarget_ID\tY\n")

        edges = set()
        for i in range(edges):
            drug = np.random.choice(drugs)
            prot = np.random.choice(prots)
            while (drug, prot) in edges:
                drug = np.random.choice(drugs)
                prot = np.random.choice(prots)

            edges.add((drug, prot))
            data.write("\t".join([drug, prot, str(np.random.normal())]) + "\n")


def assess_split(
    df: pd.DataFrame, orig_num: int, train_frac: float, val_frac: float
) -> dict:
    """Assess the split.
    Args:
        df (pd.DataFrame): DataFrame "Drug_ID", "Target_ID", "split" and "Y" columns.
        orig_num (int): Original number of interactions.
        train_frac (float, optional): value from 0 to 1, how much of the data should go into train.
        val_frac (float, optional): value from 0 to 1, how much of the data should into validation.
        Returns:
            dict: Dictionary with the following keys:
                "train_diff": Change in number of train interactions.
                "val_diff": Change in number of val interactions.
                "test_diff": Change in number of test interactions.
                "total_diff": Change in number of total interactions.
    """
    assert "split" in df.columns, "The dataframe must contain a column named 'split'"
    test_frac = 1 - train_frac - (val_frac if val_frac is not None else 0)
    train_diff = df[df["split"] == "train"].shape[0] / df.shape[0] - train_frac
    if val_frac is not None:
        val_diff = df[df["split"] == "val"].shape[0] / df.shape[0] - val_frac
    test_diff = df[df["split"] == "test"].shape[0] / df.shape[0] - test_frac
    total_diff = (df.shape[0] - orig_num) / orig_num
    if val_frac is not None:
        return dict(
            train_diff=train_diff,
            val_diff=val_diff,
            test_diff=test_diff,
            total_diff=total_diff,
        )
    else:
        return dict(train_diff=train_diff, test_diff=test_diff, total_diff=total_diff)


# register all models to be evaluated ...
models = {
    # ... for datasets to be split into two parts ...
    "split_two": {
        "genetic": GeneticSplitter,
        "louvain": LouvainSplitter,
    },
    # ... and for datasets to be split into three parts.
    "split_three": {
        "louvain": LouvainSplitter,
    },
}


# register all datasets to be used to evaluate the performance of the models
datasets = {
    # "random_100x1000": (Path(__file__).parent / "data" / "random_100x1000.tsv"),
    "glylec": (Path(__file__).parent / "data" / "glylec.tsv"),
    "glass_all": (Path(__file__).parent / "data" / "glass_all.tsv"),
    "glass_posneg": (Path(__file__).parent / "data" / "glass_posneg.tsv"),
}


def run(d, fun, val_split):
    results = []
    for i in range(1):
        print(f"\rRun {i + 1}/{1}", end="")
        metrics = {}
        df = pd.read_csv(datasets[d], sep="\t")
        orig_num = df.shape[0]
        start = time.time()
        df = fun(df)
        metrics["time"] = time.time() - start
        metrics.update(assess_split(df, orig_num, 0.7, val_split))
        results.append(metrics)
    print("\rDone")
    return pd.DataFrame(results)


def run_all():
    """Run all combinations of model configurations and datasets to find the best one."""
    # First, evaluate the models registered for splitting in two parts.
    output = []

    for model in models["split_two"].keys():
        for d in datasets.keys():
            for i, m in enumerate(models["split_two"][model].get_all()):
                print(f"Split-2: Model: {m.name} on dataset {d}")
                results = run(d, lambda x: m.split_two(x, 0.7), None)
                output.append(
                    {
                        "model": m.name,
                        "author": m.author,
                        "data": d,
                        "time": results["time"].mean(),
                        "time_std": results["time"].std(),
                        "dTrain": results["train_diff"].mean(),
                        "dTrain_std": results["train_diff"].std(),
                        "dTest": results["test_diff"].mean(),
                        "dTest_std": results["test_diff"].std(),
                        "dTotal": results["total_diff"].mean(),
                        "dTotal_std": results["total_diff"].std(),
                        "params": i,
                    }
                )
    if output:
        df = pd.DataFrame(output)
        df.sort_values(
            by=["data", "dTrain"],
            key=lambda col: col.map(lambda x: str(x) if str(x)[0] != "-" else str(x)[1:]),
            inplace=True
        )
        df.to_csv("results_split_two.tsv", sep="\t", index=False, float_format="%.4f")

    # Then, evaluate the models registered for splitting into three parts.
    output = []
    for model in models["split_three"].keys():
        for d in datasets.keys():
            for i, m in enumerate(models["split_three"][model].get_all()):
                print(f"Split-2: Model: {m.name} on dataset {d}")
                results = run(d, lambda x: m.split_three(x, 0.7, 0.2), 0.2)
                output.append(
                    {
                        "model": m.name,
                        "author": m.author,
                        "data": d,
                        "time": results["time"].mean(),
                        "time_std": results["time"].std(),
                        "dTrain": results["train_diff"].mean(),
                        "dTrain_std": results["train_diff"].std(),
                        "dVal": results["val_diff"].mean(),
                        "dVal_std": results["val_diff"].std(),
                        "dTest": results["test_diff"].mean(),
                        "dTest_std": results["test_diff"].std(),
                        "dTotal": results["total_diff"].mean(),
                        "dTotal_std": results["total_diff"].std(),
                        "params": i,
                    }
                )
    if output:
        df = pd.DataFrame(output)
        df.sort_values(
            by=["data", "dTrain"],
            key=lambda col: col.map(lambda x: str(x) if str(x)[0] != "-" else str(x)[1:]),
            inplace=True
        )
        df.to_csv("results_split_three.tsv", sep="\t", index=False, float_format="%.4f")


if __name__ == "__main__":
    run_all()
