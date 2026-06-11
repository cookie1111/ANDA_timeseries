import pytest
import torch

from ANDA import utils, anda


_NETWORK_HINTS = (
    "connection",
    "timed out",
    "temporary failure in name resolution",
    "name or service not known",
    "no route to host",
    "max retries exceeded",
    "all download strategies failed",
    "http error 403",
    "http error 404",
    "http error 5",
)


def _maybe_skip_on_network_error(exc: BaseException) -> None:
    msg = str(exc).lower()
    if any(hint in msg for hint in _NETWORK_HINTS):
        pytest.skip(f"UCR mirror unreachable from this environment: {exc}")


class TestUCRParser:
    def test_single_dataset_name(self):
        assert utils._parse_ucr_selector("UCR:ECG200") == ["ECG200"]

    def test_all_keyword_returns_full_list(self):
        names = utils._parse_ucr_selector("UCR:ALL")
        assert names == list(utils.UCR_DATASETS)
        assert len(names) == 128

    def test_explicit_list(self):
        assert utils._parse_ucr_selector("UCR:[ECG200, Adiac, FordA]") == [
            "ECG200",
            "Adiac",
            "FordA",
        ]

    def test_list_tolerates_whitespace(self):
        assert utils._parse_ucr_selector("UCR:[ ECG200 ,Adiac ]") == ["ECG200", "Adiac"]

    def test_empty_selector_raises(self):
        with pytest.raises(ValueError):
            utils._parse_ucr_selector("UCR:")

    def test_empty_list_raises(self):
        with pytest.raises(ValueError):
            utils._parse_ucr_selector("UCR:[]")


class TestGarbageInput:
    def test_unknown_dataset_name_raises_value_error(self):
        with pytest.raises(ValueError, match="not supported"):
            utils.load_full_datasets("definitely_not_a_dataset")

    def test_ucr_prefix_only_raises_value_error(self):
        with pytest.raises(ValueError):
            utils.load_full_datasets("UCR:")

    def test_ucr_empty_list_raises_value_error(self):
        with pytest.raises(ValueError):
            utils.load_full_datasets("UCR:[]")


class TestSplitGates:
    def test_load_split_datasets_rejects_ucr(self):
        with pytest.raises(NotImplementedError, match="UCR"):
            anda.load_split_datasets(dataset_name="UCR:ECG200")

    def test_load_split_datasets_dynamic_rejects_ucr(self):
        with pytest.raises(NotImplementedError, match="not supported"):
            anda.load_split_datasets_dynamic(dataset_name="UCR:ECG200")


pytest.importorskip("sktime")


class TestUnknownNameHint:
    '''_unknown_ucr_hint is pure (no network) - it turns a typo'd dataset
    name into an actionable suggestion instead of the old misleading
    "variable-length not supported" message.'''

    def test_known_name_gets_no_hint(self):
        assert utils._unknown_ucr_hint("ECG200") == ""

    def test_typo_suggests_close_matches(self):
        hint = utils._unknown_ucr_hint("ECG2000")
        assert "Did you mean" in hint
        assert "ECG" in hint

    def test_unknown_name_mentions_registry_and_uea(self):
        # CardiacArrhythmia is not a real archive dataset; the hint must say
        # so and still acknowledge UEA names like EigenWorms are valid.
        hint = utils._unknown_ucr_hint("CardiacArrhythmia")
        assert "not in the known UCR 2018" in hint
        assert "EigenWorms" in hint

    def test_uea_name_not_in_registry_still_hinted(self):
        # EigenWorms is a valid UEA dataset but absent from UCR_DATASETS, so
        # it gets a hint - we must NOT hard-reject it (the loader still tries).
        assert "EigenWorms" not in utils.UCR_DATASETS
        assert utils._unknown_ucr_hint("EigenWorms") != ""


class TestRandomUCRName:
    def test_unknown_ucr_name_raises_runtime_error(self):
        with pytest.raises(RuntimeError, match="Failed to load UCR"):
            utils.load_full_datasets("UCR:NotARealDataset_XYZ")


class TestSingleUCRLoad:
    def test_loads_ecg200(self):
        try:
            result = utils.load_full_datasets("UCR:ECG200")
        except RuntimeError as e:
            _maybe_skip_on_network_error(e)
            raise

        assert isinstance(result, list)
        assert len(result) == 4

        train_x, train_y, test_x, test_y = result
        assert isinstance(train_x, torch.Tensor)
        assert isinstance(train_y, torch.Tensor)
        assert isinstance(test_x, torch.Tensor)
        assert isinstance(test_y, torch.Tensor)

        # UCR features come back as (N, C, T) and labels as integer-encoded.
        assert train_x.dim() == 3
        assert test_x.dim() == 3
        assert train_x.shape[0] == train_y.shape[0]
        assert test_x.shape[0] == test_y.shape[0]
        assert train_y.dtype == torch.int64
        assert test_y.dtype == torch.int64

        all_y = torch.cat([train_y, test_y])
        assert all_y.min().item() == 0
        assert all_y.max().item() == all_y.unique().numel() - 1


class TestMultiUCRLoad:
    def test_multiple_correct_returns_dict_keyed_by_name(self):
        try:
            result = utils.load_full_datasets("UCR:[ECG200, Coffee]")
        except RuntimeError as e:
            _maybe_skip_on_network_error(e)
            raise

        assert isinstance(result, dict)
        assert set(result.keys()) == {"ECG200", "Coffee"}

        for four in result.values():
            assert isinstance(four, list)
            assert len(four) == 4
            train_x, train_y, test_x, test_y = four
            assert train_x.shape[0] == train_y.shape[0]
            assert test_x.shape[0] == test_y.shape[0]
            assert train_x.dim() == 3

    def test_mix_of_correct_and_wrong_raises_runtime_error(self):
        with pytest.raises(RuntimeError, match="Failed to load UCR"):
            utils.load_full_datasets("UCR:[ECG200, NotARealDataset_XYZ]")
