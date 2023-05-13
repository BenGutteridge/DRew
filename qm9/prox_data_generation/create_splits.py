import json
import pickle
from pathlib import Path
import os.path as osp
import random


if __name__ == "__main__":
    for clf_threshold in [1, 3, 5, 8, 10]:
        raw_dataset_path = osp.abspath(
            osp.join(
                osp.dirname(__file__),
                "..",
                "data",
                "Prox",
                str(clf_threshold) + "-Prox",
                "raw",
            )
        )
        with open(osp.join(raw_dataset_path, "data_list.pickle"), "rb") as f:
            graph_list = pickle.load(f)

        splits_json = []
        splits_path = osp.abspath(
            osp.join(
                osp.dirname(__file__),
                "..",
                "data_splits",
                "Prox",
                str(clf_threshold) + "-Prox" + "_splits.json",
            )
        )
        Path(
            osp.abspath(osp.join(osp.dirname(__file__), "..", "data_splits", "Prox"))
        ).mkdir(parents=True, exist_ok=True)

        # 10-CV with 10% validation
        indices = [*range(len(graph_list))]
        for i in range(10):
            test = indices[
                (i * len(graph_list) // 10) : ((i + 1) * len(graph_list) // 10)
            ]
            train = (
                indices[: (i * len(graph_list) // 10)]
                + indices[((i + 1) * len(graph_list)) // 10 :]
            )
            random.shuffle(train)

            validation_spl = len(train) // 10
            validation = train[:validation_spl]
            train = train[validation_spl:]

            current_split = {
                "test": test,
                "model_selection": [{"train": train, "validation": validation}],
            }
            splits_json.append(current_split)

        with open(splits_path, "w") as f:
            json.dump(splits_json, f)
