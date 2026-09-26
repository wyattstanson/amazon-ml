import numpy as np
import scipy.sparse as sp

from ber.blocking import _chunks_by_flops, _to_matrix, _topk_rows, sorted_lookup


def test_sorted_lookup():
    vocab = np.array([3, 7, 9, 20], dtype=np.uint32)
    ids = np.array([9, 4, 3, 20, 20, 1], dtype=np.uint32)
    assert sorted_lookup(vocab, ids).tolist() == [2, -1, 0, 3, 3, -1]


def test_to_matrix_drops_missing_columns():
    cols = np.array([0, -1, 2, 1, -1])
    vals = np.array([1.0, 9.0, 2.0, 3.0, 9.0])
    indptr = np.array([0, 2, 2, 5])
    m = _to_matrix(cols, vals, indptr, 3).toarray()
    assert m.tolist() == [[1, 0, 0], [0, 0, 0], [0, 3, 2]]


def test_topk_matches_bruteforce_with_deterministic_ties():
    rng = np.random.default_rng(0)
    dense = rng.integers(0, 4, size=(60, 40)).astype(np.float32)   # many ties
    dense[rng.random(dense.shape) < 0.5] = 0
    m = sp.csr_matrix(dense)
    for k in (1, 3, 5, 50):
        rows, cols, score, rank = _topk_rows(m.copy(), k, all_pairs=False)
        for i in range(dense.shape[0]):
            nz = [(-dense[i, j], j) for j in range(dense.shape[1]) if dense[i, j] != 0]
            expect = [j for _, j in sorted(nz)[:k]]
            got = cols[rows == i].tolist()
            assert got == expect, (k, i)
            assert rank[rows == i].tolist() == list(range(len(expect)))


def test_topk_all_pairs_keeps_everything():
    m = sp.csr_matrix(np.array([[1, 0, 2], [0, 0, 0], [5, 5, 5]], dtype=np.float32))
    rows, cols, _, _ = _topk_rows(m, 1, all_pairs=True)
    assert len(rows) == 5


def test_chunks_by_flops_cover_all_rows():
    flops = np.array([5, 5, 50, 1, 1, 1, 30], dtype=float)
    chunks = _chunks_by_flops(flops, 10)
    assert chunks[0][0] == 0 and chunks[-1][1] == len(flops)
    assert all(a[1] == b[0] for a, b in zip(chunks, chunks[1:]))
    for s, e in chunks:
        assert e - s == 1 or flops[s:e].sum() <= 10


def _tiny_table():
    """A 1-S1 / several-S2/S3 universe built through the real normalisation + tokenisation."""
    from ber.normalize import normalize_address, normalize_name
    from ber.store import RecordTable, StrCol
    from ber.tokens import TOKEN_TYPES, TokenCSR, _tokenize_chunk
    recs = [
        ("S1-1", 1, "Herrera Family Office LLC", "13802 Martin Street, Andover, MN"),
        ("S2-10", 2, "HERRERA FAMILY OFFICE (LLC)", "##13802 MARTIN ST, ANDOVER, MN"),     # true
        ("S2-11", 2, "Zephyr Labs", "99 Lake Rd, Duluth, MN"),                            # noise
        ("S3-20", 3, "Herrera Family Office", ""),                                        # true, empty addr
        ("S3-21", 3, "Acme Robotics", "5 Main St, Austin, TX"),                           # noise
    ]
    cols = {k: [] for k in RecordTable.TEXT}
    empty = []
    for eid, src, name, addr in recs:
        n, a = normalize_name(name), normalize_address(addr, "US")
        for k, v in (("entity_id", eid), ("name_full", n.full), ("name_core", n.core), ("name_alias", n.alias),
                     ("legal", n.legal), ("addr_norm", a.norm), ("state", a.state), ("nums", " ".join(a.nums)),
                     ("codes", " ".join(a.codes))):
            cols[k].append(v)
        empty.append(a.empty)
    numeric = {"source": np.array([r[1] for r in recs], np.int8), "country": np.zeros(len(recs), np.int8),
               "is_domain": np.zeros(len(recs), bool), "is_indic": np.zeros(len(recs), bool),
               "addr_empty": np.array(empty), "key": np.arange(len(recs))}
    table = RecordTable({k: StrCol.from_list(v) for k, v in cols.items()}, numeric, ["US"])
    res = _tokenize_chunk((cols["name_core"], cols["name_alias"], cols["addr_norm"], cols["state"]))
    toks = {k: TokenCSR(res[k][0], res[k][1]) for k in TOKEN_TYPES}
    return table, toks


def test_block_country_end_to_end_finds_matches_including_empty_address():
    from ber.blocking import BlockingParams, run_blocking
    table, toks = _tiny_table()
    c = run_blocking(table, toks, BlockingParams(k=1, k_empty=1))
    got = set(table.entity_id.take(c.r))
    assert {"S2-10", "S3-20"} <= got          # the empty-address true match has its own channel
    assert (c.q == 0).all()


def test_support_features_count_agreeing_candidates():
    from ber.blocking import Candidates
    from ber.features import support_features
    from ber.store import RecordTable, StrCol
    # S1 row 0 (code 100); candidates: two siblings at 102 (agree with each other), one at 100
    codes = ["100", "102 5", "102", "100", ""]
    names = ["acme", "acme", "acme", "acme co", "acme"]
    text = {k: StrCol.from_list([""] * 5) for k in RecordTable.TEXT}
    text["codes"] = StrCol.from_list(codes)
    text["name_core"] = StrCol.from_list(names)
    table = RecordTable(text, {"source": np.array([1, 2, 2, 3, 3], np.int8)}, ["US"])
    c = Candidates(np.zeros(4, np.int32), np.array([1, 2, 3, 4], np.int32), np.ones(4, np.float32),
                   np.zeros(4, np.int16))
    f = support_features(table, c)
    assert f["code_support"][:3].tolist() == [1, 1, 0] and np.isnan(f["code_support"][3])
    assert (f["code_support_s1"] == 1).all()          # one candidate shares the S1's own code
    assert f["name_support"].tolist() == [2, 2, 0, 2]
