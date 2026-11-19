from itertools import product
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rwt.offset_path import minimum_cost_offset_path


def read_cost(token):
    return np.inf if token == "x" else float(token)


def cases():
    for line in (ROOT / "qa/path_vectors.txt").read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        fields = iter(line.split())
        name, rows, bins = next(fields), int(next(fields)), int(next(fields))
        stations = np.array([read_cost(next(fields)) for _ in range(rows*bins)]).reshape(rows, bins)
        mode = next(fields)
        assert mode in ("free", "edges")
        edges = None
        if mode == "edges":
            count = max(0, rows-1)
            edges = np.array([read_cost(next(fields)) for _ in range(count*3*bins)]).reshape(count, 3, bins)
        total = read_cost(next(fields))
        expected = [int(x) for x in fields]
        assert len(expected) == (rows if np.isfinite(total) else 0)
        yield name, stations, edges, total, expected


def test_production_path_and_cost_match_frozen_vectors():
    names = set()
    for name, stations, edges, total, expected in cases():
        assert name not in names
        names.add(name)
        if np.isinf(total):
            with pytest.raises(ValueError, match="no reachable finite-cost path"):
                minimum_cost_offset_path(stations, edges)
            continue
        path = minimum_cost_offset_path(stations, edges)
        assert list(path) == expected, name
        recovered = sum(stations[t, j] for t, j in enumerate(path))
        if edges is not None:
            recovered += sum(edges[t, 1+b-a, b] for t, (a,b) in enumerate(zip(path, path[1:])))
        assert recovered == total, name
    assert len(names) == 1309


def test_signed_station_and_edge_mask_sweep_is_exhaustive():
    observed = {}
    for name, stations, edges, _, _ in cases():
        if name.startswith("sweep_"):
            assert stations.shape == (2,2) and edges.shape == (1,3,2)
            mask = sum(1 << (2*src + dst) for src,dst in product(range(2), repeat=2)
                       if np.isfinite(edges[0, 1+dst-src, dst]))
            identity = (tuple(stations.flat), mask)
            assert identity not in observed
            observed[identity] = name
    assert set(observed) == set(product(product((-1,0,1), repeat=4), range(16)))
