import json
import os
import random
from typing import Any, Dict, List, Optional, Tuple


# to execute this class from shell: python -c "from utils import TacredSampler; TacredSampler(seed=42).sample_splits_stratified('data/TACRED')"

class TacredSampler:
    """Create smaller TACRED JSON splits by random sampling.

    Works with the common TACRED flattened schema, e.g.:
      - token (list[str])
      - subj_start/subj_end/obj_start/obj_end
      - subj_type/obj_type
      - relation

    But sampling is schema-agnostic: we simply read a JSON list and write a subset.
    """

    def __init__(self, seed: int = 42):
        self.seed = seed

    @staticmethod
    def _read_json_list(path: str) -> List[Dict[str, Any]]:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"Expected a JSON list in {path}, got {type(data)}")
        return data

    @staticmethod
    def _write_json_list(path: str, data: List[Dict[str, Any]]) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def sample_file(
            self,
            input_path: str,
            output_path: str,
            n: int,
            *,
            replace: bool = False,
            seed: Optional[int] = None,
            sort_by_id: bool = False,
    ) -> None:
        """Sample `n` examples from `input_path` and write them to `output_path`.

        Args:
            input_path: Path to TACRED split JSON (a list of dicts).
            output_path: Where to write the sampled JSON.
            n: Number of examples to sample.
            replace: Sample with replacement (default False).
            seed: Override the sampler seed for this call.
            sort_by_id: If True, sort the sampled subset by `id` field when present.
        """
        rng = random.Random(self.seed if seed is None else seed)
        data = self._read_json_list(input_path)

        if n <= 0:
            raise ValueError("n must be > 0")

        if not replace and n > len(data):
            raise ValueError(
                f"Requested n={n} without replacement, but dataset has only {len(data)} examples: {input_path}"
            )

        if replace:
            sampled = [rng.choice(data) for _ in range(n)]
        else:
            sampled = rng.sample(data, n)

        if sort_by_id and sampled and isinstance(sampled[0], dict) and "id" in sampled[0]:
            sampled = sorted(sampled, key=lambda x: str(x.get("id", "")))

        self._write_json_list(output_path, sampled)

    def sample_file_stratified(
            self,
            input_path: str,
            output_path: str,
            *,
            per_relation: int = 0,
            per_relation_map: Optional[Dict[str, int]] = None,
            no_relation_n: int = 0,
            seed: Optional[int] = None,
            rel_key: str = "relation",
            shuffle_output: bool = True,
    ) -> None:
        """Create a stratified sample from a TACRED JSON split.

        You can either:
          - set `per_relation` to sample that many examples for every relation != no_relation
          - OR provide `per_relation_map` to control counts per specific relation label

        Additionally, control how many `no_relation` examples to include via `no_relation_n`.

        Notes:
          - Sampling is WITHOUT replacement.
          - If requested count exceeds available, we take all available for that label.

        Args:
            input_path: Path to TACRED split JSON (list of dicts).
            output_path: Where to write the sampled JSON.
            per_relation: Count for each positive relation (ignored if per_relation_map provided).
            per_relation_map: Optional dict mapping relation -> count.
            no_relation_n: Count for no_relation.
            seed: RNG seed.
            rel_key: Field name for the label (default: "relation").
            shuffle_output: Shuffle final sampled list before writing.
        """
        rng = random.Random(self.seed if seed is None else seed)
        data = self._read_json_list(input_path)
        groups = self._group_by_relation(data, rel_key=rel_key)

        sampled: List[Dict[str, Any]] = []

        # 1) no_relation
        nr_pool = groups.get("no_relation", [])
        if no_relation_n > 0:
            k = min(no_relation_n, len(nr_pool))
            sampled.extend(rng.sample(nr_pool, k))

        # 2) positive relations
        if per_relation_map is not None:
            # Explicit per-label counts
            for rel, n in per_relation_map.items():
                if n <= 0:
                    continue
                pool = groups.get(rel, [])
                k = min(n, len(pool))
                if k > 0:
                    sampled.extend(rng.sample(pool, k))
        else:
            # Uniform count for every positive relation
            if per_relation <= 0:
                raise ValueError("Set per_relation > 0 or provide per_relation_map.")
            for rel, pool in groups.items():
                if rel == "no_relation":
                    continue
                k = min(per_relation, len(pool))
                if k > 0:
                    sampled.extend(rng.sample(pool, k))

        if shuffle_output:
            rng.shuffle(sampled)

        self._write_json_list(output_path, sampled)

    def sample_splits(
            self,
            data_dir: str,
            n_train: int,
            n_dev: int,
            n_test: int,
            *,
            train_in: str = "train.json",
            dev_in: str = "dev.json",
            test_in: str = "test.json",
            train_out: str = "train_sample.json",
            dev_out: str = "dev_sample.json",
            test_out: str = "test_sample.json",
            seed: Optional[int] = None,
    ) -> None:
        """Convenience wrapper to sample train/dev/test in one call."""
        self.sample_file(
            os.path.join(data_dir, train_in),
            os.path.join(data_dir, train_out),
            n_train,
            seed=seed,
        )
        self.sample_file(
            os.path.join(data_dir, dev_in),
            os.path.join(data_dir, dev_out),
            n_dev,
            seed=None if seed is None else seed + 1,
        )
        self.sample_file(
            os.path.join(data_dir, test_in),
            os.path.join(data_dir, test_out),
            n_test,
            seed=None if seed is None else seed + 2,
        )

    def sample_splits_stratified(
            self,
            data_dir: str,
            *,
            # Train
            train_per_relation: int = 20,
            train_no_relation_n: int = 1300,
            train_per_relation_map: Optional[Dict[str, int]] = None,
            # Dev
            dev_per_relation: int = 50,
            dev_no_relation_n: int = 2000,
            dev_per_relation_map: Optional[Dict[str, int]] = None,
            # Test
            test_per_relation: int = 0,
            test_no_relation_n: int = 0,
            test_per_relation_map: Optional[Dict[str, int]] = None,
            # Filenames
            train_in: str = "train.json",
            dev_in: str = "dev.json",
            test_in: str = "test.json",
            train_out: str = "train_sample.json",
            dev_out: str = "dev_sample.json",
            test_out: str = "test_sample.json",
            seed: Optional[int] = None,
            rel_key: str = "relation",
    ) -> None:
        """Create stratified samples for train/dev/test with controllable relation counts.

        Typical debug recipe:
          - small, balanced-ish positives (e.g., 10 per positive relation)
          - a capped amount of no_relation (e.g., 200)

        For test, you can leave counts at 0 to skip creating a test sample.
        """
        base_seed = self.seed if seed is None else seed

        self.sample_file_stratified(
            os.path.join(data_dir, train_in),
            os.path.join(data_dir, train_out),
            per_relation=train_per_relation,
            per_relation_map=train_per_relation_map,
            no_relation_n=train_no_relation_n,
            seed=base_seed,
            rel_key=rel_key,
        )

        self.sample_file_stratified(
            os.path.join(data_dir, dev_in),
            os.path.join(data_dir, dev_out),
            per_relation=dev_per_relation,
            per_relation_map=dev_per_relation_map,
            no_relation_n=dev_no_relation_n,
            seed=base_seed + 1,
            rel_key=rel_key,
        )

        # Only create test sample if requested
        if (test_per_relation_map is not None) or (test_per_relation > 0) or (test_no_relation_n > 0):
            self.sample_file_stratified(
                os.path.join(data_dir, test_in),
                os.path.join(data_dir, test_out),
                per_relation=test_per_relation,
                per_relation_map=test_per_relation_map,
                no_relation_n=test_no_relation_n,
                seed=base_seed + 2,
                rel_key=rel_key,
            )

    @staticmethod
    def _group_by_relation(data: List[Dict[str, Any]], rel_key: str = "relation") -> Dict[str, List[Dict[str, Any]]]:
        groups: Dict[str, List[Dict[str, Any]]] = {}
        for ex in data:
            rel = ex.get(rel_key)
            if rel is None:
                raise ValueError(f"Example missing '{rel_key}' field. Keys={list(ex.keys())}")
            groups.setdefault(str(rel), []).append(ex)
        return groups
