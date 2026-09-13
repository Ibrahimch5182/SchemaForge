from localsql.data.splitter import should_keep_business_context, split_database_ids


def test_split_is_deterministic():
    db_ids = [f"db_{i}" for i in range(30)]
    train_a, val_a = split_database_ids(db_ids, seed=42, train_fraction=0.9)
    train_b, val_b = split_database_ids(db_ids, seed=42, train_fraction=0.9)
    assert train_a == train_b
    assert val_a == val_b


def test_split_has_zero_overlap():
    db_ids = [f"db_{i}" for i in range(50)]
    train_ids, val_ids = split_database_ids(db_ids, seed=42, train_fraction=0.9)
    assert train_ids & val_ids == set()


def test_split_assigns_every_db_exactly_once():
    db_ids = [f"db_{i}" for i in range(50)]
    train_ids, val_ids = split_database_ids(db_ids, seed=42, train_fraction=0.9)
    assert train_ids | val_ids == set(db_ids)
    assert len(train_ids) + len(val_ids) == len(set(db_ids))


def test_split_different_seeds_can_differ():
    db_ids = [f"db_{i}" for i in range(50)]
    train_a, _ = split_database_ids(db_ids, seed=42, train_fraction=0.9)
    train_b, _ = split_database_ids(db_ids, seed=7, train_fraction=0.9)
    assert train_a != train_b


def test_evidence_dropout_is_deterministic():
    decisions_a = [should_keep_business_context(f"db:{i}", seed=42) for i in range(200)]
    decisions_b = [should_keep_business_context(f"db:{i}", seed=42) for i in range(200)]
    assert decisions_a == decisions_b


def test_evidence_dropout_roughly_matches_probability():
    kept = sum(should_keep_business_context(f"db:{i}", seed=42, keep_probability=0.5) for i in range(2000))
    fraction = kept / 2000
    assert 0.4 < fraction < 0.6


def test_evidence_dropout_extremes():
    assert all(
        should_keep_business_context(f"db:{i}", seed=1, keep_probability=1.0) for i in range(100)
    )
    assert not any(
        should_keep_business_context(f"db:{i}", seed=1, keep_probability=0.0) for i in range(100)
    )
