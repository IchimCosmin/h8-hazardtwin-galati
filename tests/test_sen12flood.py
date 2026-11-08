"""Runs the pipeline end to end on a tiny synthetic dataset with the SEN12FLOOD layout."""
import json

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")
from rasterio.transform import from_origin

from h8twin.cli import main
from h8twin.sen12flood import build_index, is_linear, load_sequence, split_sequences


def _tif(path, arr):
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1,
                       dtype="float32", crs="EPSG:4326", transform=from_origin(30, 10, 1e-4, 1e-4)) as d:
        d.write(arr.astype("float32"), 1)


@pytest.fixture
def root(tmp_path):
    rng = np.random.default_rng(0)
    s1, s2 = {}, {}
    for k, (lon, lat) in enumerate([(30.1, 10.2), (30.2, 10.1), (140.0, -30.0)], start=1):
        seq = f"{k:04d}"
        (tmp_path / seq).mkdir()
        s1[seq] = {"count": 4, "folder": seq, "geo": {"type": "Point", "coordinates": [lon, lat]}}
        s2[seq] = {"count": 1, "folder": seq, "geo": {"type": "Point", "coordinates": [lon, lat]}}
        for t in range(4):
            flood = t >= 2
            pre = f"S1A_{seq}_{t}"
            s1[seq][str(t + 1)] = {"date": f"2019-01-{1 + 6 * t:02d}", "FLOODING": flood,
                                   "FULL-DATA-COVERAGE": True, "filename": pre, "orbit": "ASCENDING"}
            lin = rng.uniform(0.02, 0.2, (40, 40))
            if flood:
                lin[:20] = 0.003  # dark water
            _tif(tmp_path / seq / f"{pre}_corrected_VV.tif", lin)
            _tif(tmp_path / seq / f"{pre}_corrected_VH.tif", lin / 4)
        s2[seq]["1"] = {"date": "2019-01-03", "FLOODING": False, "FULL-DATA-COVERAGE": False, "filename": f"S2_{seq}"}
    (tmp_path / "S1list.json").write_text(json.dumps(s1))
    (tmp_path / "S2list.json").write_text(json.dumps(s2))
    return tmp_path


def test_index_and_sequence(root):
    idx = build_index(root)
    s1 = idx[idx.sensor == "S1"]
    assert len(s1) == 12 and s1.available.all()
    assert not idx[idx.sensor == "S2"].available.any()  # S2 bands absent in the fixture
    s = load_sequence(idx, "0001", size=32)
    assert s.x.shape == (4, 2, 32, 32)
    assert s.y.tolist() == [0, 0, 1, 1]
    assert not is_linear(s.x)  # converted to dB
    assert np.nanmean(s.x[3, 0, :10]) < -20  # water is dark


def test_split_keeps_regions_together(root):
    idx = build_index(root)
    sp = split_sequences(idx[idx.sensor == "S1"], test_frac=0.34, seed=1)
    by_group = sp.groupby("group").split.nunique()
    assert (by_group == 1).all()
    assert set(sp.split) == {"train", "test"}


def test_cli(root, tmp_path):
    out = tmp_path / "out"
    main(["ingest", "--root", str(root), "--out", str(out), "--n-pixels", "5"])
    st = json.loads((out / "stats.json").read_text())
    assert st["index"]["S1"]["images_listed"] == 12
    assert st["index"]["S1"]["revisit_days_median"] == 6
    main(["figure", "--out", str(out)])
    assert (out / "fig_cp1_sequence.png").exists()


def test_learn_runs(root, tmp_path):
    from h8twin.features import extract_features
    from h8twin.learn import METHODS, otsu_from_hist, run_experiment

    idx = build_index(root, sensors=("S1",))
    df, hists = extract_features(idx, n_jobs=1)
    assert len(df) == 12 and hists.shape == (12, 180)
    assert df.loc[df.t_index == 0, "n_prev"].eq(0).all()
    # dark water appears after the label -> VV drops vs. the past
    assert df.loc[df.flooding == 1, "d_vv_lt3"].min() > 0.3
    thr = otsu_from_hist(hists.sum(0))
    assert -35 < thr < 0
    res = run_experiment(df, hists, seeds=(0, 1), test_frac=0.34)
    runs = res[res.row_type == "run"]
    assert set(runs.method) == set(METHODS)
    assert {"summary", "paired_diff"} <= set(res.row_type)
