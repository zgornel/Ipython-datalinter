"""Tests for ``datalintermagic.datalintermagic``.

Regenerated to match the current source on the cc-development branch.

Maintainability notes:
- Message fragments are centralized and match the wording in the source exactly
  (including the typo "Ivalid").
- Warnings are emitted via warnings.warn(..., category=DataLinterWarning).
- Display output uses IPython.display.display + Pretty + termcolor.colored.
- Parsing errors return None and emit a warning (they do not raise).
- Only valid Python identifiers are accepted for tracked variables.
- Cell magic always executes the cell, even if parsing or linting fails.
- HTTP connection timeout is always DEFAULT_TIMEOUT (the timeout argument is
  only used in the timeout error message).
"""

import json
import socket
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure local src is importable when running from a checkout
_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from datalintermagic.datalintermagic import (  # noqa: E402
    DEFAULT_DELIMITER,
    DEFAULT_IP,
    DEFAULT_PORT,
    DEFAULT_TIMEOUT,
    DataLinterMagic,
    DataLinterWarning,
    LinterConnectionError,
    LinterHTTPError,
    _parse_delimiter,
)

# ---------------------------------------------------------------------------
# Centralized expected message fragments (update here if source wording changes)
# ---------------------------------------------------------------------------
MSG_MISSING_TRACKED = "Use '--tracked-variable' to specify a data variable!"
MSG_INVALID_IDENTIFIER = "Ivalid identifier specified with '--tracked-variable'"
MSG_ADD_PARSE_ERROR = "Error in %add magic line parser:"
MSG_LINE_PARSE_ERROR = "Line parse error:"
MSG_ADD_FAIL = "Could not add"
MSG_ADD_PARSE_FAIL = "Could not parse line for '%add' magic"
MSG_LINTLINE_NEEDS_SYMBOL = "'%lintline' magic needs a valid data symbol"
MSG_LINTLINE_PARSE_FAIL = "Could not parse line for '%lintline' magic"
MSG_LINTCELL_PARSE_FAIL = "Could not parse line for '%lintcell' magic"
MSG_CANNOT_REACH = "Cannot reach server:"
MSG_HTTP_REJECT = "Server rejected the request"
MSG_FAILED_READ = "Failed to read linter output"
MSG_FAILED_READ_CELL = "Failed to read linter output, perhaps linting failed"
MSG_GENERATED_NAMES = "will be auto-generated as x0,x1..."
MSG_OUTPUT_HEADER = "DataLinter:"

# Small reusable CSV fixtures
TWO_COL_NO_HEADER = "1,3\n2,4\n"
TWO_COL_HEADER = "a,b\n1,3\n2,4\n"
TWO_COL_PIPE_HEADER = "a|b\n1|3\n2|4\n"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def shell():
    import IPython
    from IPython.core.interactiveshell import InteractiveShell

    ipython = InteractiveShell.instance()
    assert IPython.get_ipython() is ipython
    return ipython


@pytest.fixture
def magic(shell):
    m = DataLinterMagic(shell)
    m.all_tracked_variables.clear()
    return m


@pytest.fixture
def two_col_frame():
    return pd.DataFrame({"a": [1, 2], "b": [3, 4]})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def capture_display(monkeypatch):
    """Monkeypatch display() to collect Pretty (or other) objects as strings.

    ANSI colour codes from termcolor are stripped so assertions stay simple.
    """
    displayed = []

    def fake_display(obj, *args, **kwargs):
        text = getattr(obj, "data", None)
        if text is None:
            text = str(obj)
        # Strip common ANSI escape sequences
        import re
        text = re.sub(r"\x1b\[[0-9;]*m", "", text)
        displayed.append(text)

    monkeypatch.setattr("datalintermagic.datalintermagic.display", fake_display)
    return displayed


def capture_warnings():
    """Context manager that records warning messages of category DataLinterWarning."""
    return warnings.catch_warnings(record=True)


def assert_any_warning_contains(warning_list, fragment):
    messages = [str(w.message) for w in warning_list if issubclass(w.category, DataLinterWarning)]
    assert any(fragment in msg for msg in messages), (
        f"No warning containing {fragment!r}. Got: {messages}"
    )


def assert_any_display_contains(displayed, fragment):
    assert any(fragment in text for text in displayed), (
        f"No display containing {fragment!r}. Got: {displayed}"
    )


def _install_http(monkeypatch):
    """Replace HTTPConnection with a recording fake. Returns a mutable state dict."""
    state = {
        "response": None,
        "request_error": None,
        "response_error": None,
        "conn": None,
    }

    class FakeResponse:
        def __init__(self, status, reason, body, content_type="application/json"):
            self.status = status
            self.reason = reason
            self._body = body if isinstance(body, bytes) else body.encode("utf-8")
            self._content_type = content_type

        def read(self):
            return self._body

        def getheader(self, name, default=None):
            if str(name).lower() == "content-type":
                return self._content_type
            return default

    class FakeConn:
        def __init__(
            self,
            host,
            port=None,
            timeout=socket._GLOBAL_DEFAULT_TIMEOUT,
            source_address=None,
            blocksize=8192,
        ):
            self.host = host
            self.port = port
            self.timeout = timeout
            self.closed = False
            self.requested = None
            state["conn"] = self

        def request(self, method, url, body=None, headers=None, *, encode_chunked=False):
            self.requested = {
                "method": method,
                "url": url,
                "body": body,
                "headers": dict(headers or {}),
            }
            if state["request_error"] is not None:
                raise state["request_error"]

        def getresponse(self):
            if state["response_error"] is not None:
                raise state["response_error"]
            return state["response"]

        def close(self):
            self.closed = True

    state["FakeResponse"] = FakeResponse
    monkeypatch.setattr(
        "datalintermagic.datalintermagic.http.client.HTTPConnection",
        FakeConn,
    )
    return state


def _capture_lint(monkeypatch, magic, response=None):
    """Monkeypatch http_lint_request to record the call and return a canned response."""
    if response is None:
        response = {"linting_output": "ok"}
    captured = {}

    def fake_http(ip, port, body, timeout=DEFAULT_TIMEOUT, path="/api/lint"):
        captured["ip"] = ip
        captured["port"] = port
        captured["body"] = body
        captured["timeout"] = timeout
        captured["path"] = path
        return response

    monkeypatch.setattr(magic, "http_lint_request", fake_http)
    return captured


def make_line_payload(data, delim=",", header=False, show_stats=False,
                      show_passing=False, show_na=False):
    return {
        "linter_input": {
            "context": {
                "data": data,
                "data_delim": delim,
                "data_header": header,
                "code": "",
            },
            "options": {
                "show_stats": show_stats,
                "show_passing": show_passing,
                "show_na": show_na,
            },
        }
    }


def make_cell_payload(tracked, cell, show_stats=False, show_passing=False, show_na=False):
    # tracked values are tuples; JSON will turn them into lists
    return {
        "linter_input": {
            "context": {
                "data": tracked,
                "code": cell,
            },
            "options": {
                "show_stats": show_stats,
                "show_passing": show_passing,
                "show_na": show_na,
            },
        }
    }


def assert_payload_equals(body_bytes, expected_dict):
    assert json.loads(body_bytes.decode()) == expected_dict


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestExceptions:
    def test_connection_error_message(self):
        err = LinterConnectionError("cannot connect")
        assert isinstance(err, Exception)
        assert str(err) == "cannot connect"

    def test_http_error_keeps_status_reason_and_body(self):
        err = LinterHTTPError(503, "Service Unavailable", "overloaded")
        assert isinstance(err, Exception)
        assert err.status == 503
        assert err.reason == "Service Unavailable"
        assert err.body == "overloaded"
        assert str(err) == "overloaded"

    def test_datalinter_warning_is_userwarning(self):
        assert issubclass(DataLinterWarning, UserWarning)


class TestDefaults:
    def test_constant_values(self):
        assert DEFAULT_IP == "127.0.0.1"
        assert DEFAULT_PORT == 10000
        assert DEFAULT_TIMEOUT == 300
        assert DEFAULT_DELIMITER == ","

    def test_magic_names_are_registered(self):
        assert DataLinterMagic.magics["line"]["add"] == "add"
        assert DataLinterMagic.magics["line"]["lintline"] == "lintline"
        assert DataLinterMagic.magics["cell"]["lintcell"] == "lintcell"


class TestParseDelimiter:
    def test_none_uses_default(self):
        assert _parse_delimiter(None) == ","
        assert _parse_delimiter(None, default="|") == "|"

    def test_string_is_returned_unchanged(self):
        assert _parse_delimiter(";") == ";"
        assert _parse_delimiter("") == ""
        assert _parse_delimiter(" ; ") == " ; "

    @pytest.mark.parametrize("value", [5, b",", 1.5, ("a",)])
    def test_non_string_is_rejected(self, value):
        with pytest.raises(ValueError, match="expected str"):
            _parse_delimiter(value)


class TestParseAddMagic:
    @pytest.mark.parametrize(
        "line,expected",
        [
            ("--tracked-variable df", ("df", False, ",")),
            ("--tracked-variable=df", ("df", False, ",")),
            (
                "--data-delim ; --tracked-variable df --data-header",
                ("df", True, ";"),
            ),
            ("--tracked-variable df --data-delim ''", ("df", False, "")),
        ],
    )
    def test_success_valid_identifier(self, magic, line, expected):
        with capture_warnings() as rec:
            result = magic._parse_add_magic(line)
        assert result == expected
        assert not any(issubclass(w.category, DataLinterWarning) for w in rec)

    def test_missing_tracked_variable(self, magic):
        with capture_warnings() as rec:
            result = magic._parse_add_magic("")
        assert result is None
        assert_any_warning_contains(rec, MSG_MISSING_TRACKED)

    def test_invalid_identifier(self, magic):
        with capture_warnings() as rec:
            result = magic._parse_add_magic("--tracked-variable 123bad")
        assert result is None
        assert_any_warning_contains(rec, MSG_INVALID_IDENTIFIER)

    def test_unknown_argument_raises_systemexit(self, magic):
        # argparse still raises SystemExit for unrecognized arguments even with
        # exit_on_error=False in some Python versions; SystemExit is not caught
        # by the bare `except Exception` in the source.
        result = magic._parse_add_magic("--unknown foo")
        assert result is None


class TestParseLintMagic:
    def test_defaults_when_empty(self, magic):
        with capture_warnings() as rec:
            result = magic._parse_lint_magic("")
        assert result == (None, DEFAULT_IP, DEFAULT_PORT, False, False, False)
        assert not any(issubclass(w.category, DataLinterWarning) for w in rec)

    def test_positional_and_flags(self, magic):
        result = magic._parse_lint_magic("myvar --show-stats --show-na --ip 10.0.0.1 --port 1234")
        assert result == ("myvar", "10.0.0.1", 1234, True, True, False)

    def test_show_passing(self, magic):
        result = magic._parse_lint_magic("--show-passing")
        assert result[4] is False  # show_na
        assert result[3] is False  # show_stats
        assert result[5] is True   # show_passing

    def test_invalid_ip(self, magic):
        with capture_warnings() as rec:
            result = magic._parse_lint_magic("--ip not-an-ip")
        assert result is None
        assert_any_warning_contains(rec, MSG_LINE_PARSE_ERROR)
        assert_any_warning_contains(rec, "not valid")

    def test_invalid_port(self, magic):
        with capture_warnings() as rec:
            result = magic._parse_lint_magic("--port 99999")
        assert result is None
        assert_any_warning_contains(rec, MSG_LINE_PARSE_ERROR)

    def test_non_integer_port(self, magic):
        with capture_warnings() as rec:
            result = magic._parse_lint_magic("--port abc")
        assert result is None
        assert_any_warning_contains(rec, MSG_LINE_PARSE_ERROR)


class TestCsvConversion:
    def test_dataframe_no_header(self, magic, two_col_frame):
        assert magic.dataframe_to_csv_string(two_col_frame, False, ",") == TWO_COL_NO_HEADER

    def test_dataframe_with_header(self, magic, two_col_frame):
        assert magic.dataframe_to_csv_string(two_col_frame, True, ",") == TWO_COL_HEADER

    def test_dataframe_custom_delimiter(self, magic, two_col_frame):
        assert magic.dataframe_to_csv_string(two_col_frame, True, "|") == TWO_COL_PIPE_HEADER

    def test_ndarray_conversion(self, magic):
        arr = np.array([[1, 3], [2, 4]])
        assert magic.ndarray_to_csv_string(arr, False, ",") == TWO_COL_NO_HEADER
        assert magic.ndarray_to_csv_string(arr, True, ",") == "x0,x1\n1,3\n2,4\n"

    def test_to_csv_string_dispatches(self, magic, two_col_frame):
        assert magic.to_csv_string(two_col_frame, False, ",") == TWO_COL_NO_HEADER
        arr = np.array([[10]])
        assert magic.to_csv_string(arr, False, ",") == "10\n"

    def test_unsupported_type_raises(self, magic):
        with pytest.raises(TypeError, match="Unsupported type"):
            magic.to_csv_string("not a frame", False, ",")

    def test_non_string_columns_generate_names_and_warn(self, magic):
        df = pd.DataFrame([[1, 2]], columns=[0, "b"])
        with capture_warnings() as rec:
            result = magic.dataframe_to_csv_string(df, True, ",", symbol="mix")
        assert result == "x0,x1\n1,2\n"
        assert_any_warning_contains(rec, MSG_GENERATED_NAMES)
        assert_any_warning_contains(rec, "mix")

    def test_na_representation(self, magic):
        df = pd.DataFrame({"a": [1, None]})
        result = magic.dataframe_to_csv_string(df, False, ",")
        assert "NA" in result


class TestHttpLintRequest:
    def test_successful_json_post(self, magic, monkeypatch):
        state = _install_http(monkeypatch)
        state["response"] = state["FakeResponse"](
            200, "OK", '{"linting_output": "caf\\u00e9"}'
        )
        body = "é".encode("utf-8")

        parsed = magic.http_lint_request(
            "localhost", "8080", body, timeout=3.5, path="/v2/lint"
        )

        conn = state["conn"]
        assert parsed == {"linting_output": "café"}
        assert conn.host == "localhost"
        assert conn.port == 8080
        # Connection timeout is always DEFAULT_TIMEOUT, not the argument
        assert conn.timeout == DEFAULT_TIMEOUT
        assert conn.closed is True
        assert conn.requested["method"] == "POST"
        assert conn.requested["url"] == "/v2/lint"
        assert conn.requested["body"] == body
        assert conn.requested["headers"]["Content-Type"] == "application/json"
        assert conn.requested["headers"]["Content-Length"] == str(len(body))

    def test_default_path_and_timeout_constant(self, magic, monkeypatch):
        state = _install_http(monkeypatch)
        state["response"] = state["FakeResponse"](200, "OK", b'{"ok": true}')

        assert magic.http_lint_request(DEFAULT_IP, DEFAULT_PORT, b"{}") == {"ok": True}
        assert state["conn"].timeout == DEFAULT_TIMEOUT
        assert state["conn"].requested["url"] == "/api/lint"

    @pytest.mark.parametrize("host", ["0.0.0.0", "::", "[::]"])
    def test_wildcard_hosts_connect_to_loopback(self, magic, monkeypatch, host):
        state = _install_http(monkeypatch)
        state["response"] = state["FakeResponse"](200, "OK", b"{}")

        assert magic.http_lint_request(host, 9, b"{}") == {}
        assert state["conn"].host == DEFAULT_IP
        assert state["conn"].port == 9

    def test_non_200_raises_http_error(self, magic, monkeypatch):
        state = _install_http(monkeypatch)
        state["response"] = state["FakeResponse"](
            201, "Created", '{"linting_output": "ignored"}'
        )

        with pytest.raises(LinterHTTPError) as exc:
            magic.http_lint_request("127.0.0.1", 10000, b"{}")

        assert exc.value.status == 201
        assert exc.value.reason == "Created"
        assert exc.value.body == '{"linting_output": "ignored"}'
        assert state["conn"].closed is True

    def test_invalid_json_raises_connection_error(self, magic, monkeypatch):
        state = _install_http(monkeypatch)
        state["response"] = state["FakeResponse"](200, "OK", b"not json")

        with pytest.raises(LinterConnectionError, match="Invalid JSON"):
            magic.http_lint_request("127.0.0.1", 10000, b"{}")

    def test_connection_refused_raises(self, magic, monkeypatch):
        state = _install_http(monkeypatch)
        state["request_error"] = ConnectionRefusedError("refused")

        with pytest.raises(LinterConnectionError, match="Could not reach"):
            magic.http_lint_request("127.0.0.1", 9, b"{}")

    def test_socket_timeout_uses_timeout_argument_in_message(self, magic, monkeypatch):
        state = _install_http(monkeypatch)
        state["request_error"] = socket.timeout("timed out")

        with pytest.raises(LinterConnectionError, match=r"Timed out after 7s"):
            magic.http_lint_request("127.0.0.1", 10000, b"{}", timeout=7)


class TestAddMagic:
    def test_tracks_dataframe_and_returns_none(self, magic, shell, two_col_frame):
        shell.user_ns["df_add"] = two_col_frame
        with capture_warnings() as rec:
            assert magic.add("--tracked-variable df_add") is None
        assert not any(issubclass(w.category, DataLinterWarning) for w in rec)
        csv_text, header, delim = magic.all_tracked_variables["df_add"]
        assert (csv_text, header, delim) == (TWO_COL_NO_HEADER, False, ",")

    def test_tracks_dataframe_with_header_and_custom_delim(self, magic, shell, two_col_frame):
        shell.user_ns["df_add"] = two_col_frame
        magic.add("--tracked-variable df_add --data-header --data-delim |")
        assert magic.all_tracked_variables["df_add"] == (
            TWO_COL_PIPE_HEADER,
            True,
            "|",
        )

    def test_tracks_ndarray(self, magic, shell):
        shell.user_ns["arr_add"] = np.array([[5, 6]])
        magic.add("--tracked-variable arr_add --data-header")
        csv_text, header, delim = magic.all_tracked_variables["arr_add"]
        assert header is True
        assert "x0,x1" in csv_text or "5,6" in csv_text

    def test_missing_variable_in_namespace_warns(self, magic, shell):
        with capture_warnings() as rec:
            magic.add("--tracked-variable missing_df")
        assert_any_warning_contains(rec, MSG_ADD_FAIL)
        assert "missing_df" not in magic.all_tracked_variables

    def test_unsupported_type_warns(self, magic, shell):
        shell.user_ns["bad"] = 42
        with capture_warnings() as rec:
            magic.add("--tracked-variable bad")
        assert_any_warning_contains(rec, MSG_ADD_FAIL)
        assert "bad" not in magic.all_tracked_variables

    def test_parse_failure_warns_twice(self, magic):
        # _parse_add_magic warns, then add warns again
        with capture_warnings() as rec:
            magic.add("--tracked-variable 123bad")
        msgs = [str(w.message) for w in rec if issubclass(w.category, DataLinterWarning)]
        assert any(MSG_INVALID_IDENTIFIER in m for m in msgs)
        assert any(MSG_ADD_PARSE_FAIL in m for m in msgs)


class TestLintLineMagic:
    def test_posts_tracked_frame_and_displays_output(self, magic, shell, monkeypatch, two_col_frame):
        displayed = capture_display(monkeypatch)
        shell.user_ns["df_line"] = two_col_frame
        magic.add("--tracked-variable df_line --data-header --data-delim ;")
        displayed.clear()
        captured = _capture_lint(
            monkeypatch, magic, response={"linting_output": "row 1: ok\nrow 2: ok"}
        )

        with capture_warnings() as rec:
            assert magic.lintline("df_line --show-stats --port 2222") is None

        assert not any(issubclass(w.category, DataLinterWarning) for w in rec)
        assert captured["ip"] == DEFAULT_IP
        assert captured["port"] == 2222
        assert captured["path"] == "/api/lint"
        assert_payload_equals(
            captured["body"],
            make_line_payload(
                data="a;b\n1;3\n2;4\n",
                delim=";",
                header=True,
                show_stats=True,
            ),
        )
        assert_any_display_contains(displayed, MSG_OUTPUT_HEADER)
        assert_any_display_contains(displayed, "row 1: ok")

    def test_refreshes_from_user_namespace_before_the_request(self, magic, shell, monkeypatch):
        capture_display(monkeypatch)
        shell.user_ns["df_refresh"] = pd.DataFrame({"a": [1], "b": [2]})
        magic.add("--tracked-variable df_refresh")
        shell.user_ns["df_refresh"] = pd.DataFrame({"a": [7, 8], "b": [2, 9]})
        captured = _capture_lint(monkeypatch, magic)

        magic.lintline("df_refresh --show-na --show-passing")

        assert_payload_equals(
            captured["body"],
            make_line_payload(
                data="7,2\n8,9\n",
                show_passing=True,
                show_na=True,
            ),
        )
        assert magic.all_tracked_variables["df_refresh"][0] == "7,2\n8,9\n"

    def test_keeps_cached_csv_when_name_was_deleted(self, magic, shell, monkeypatch):
        capture_display(monkeypatch)
        shell.user_ns["df_cached"] = pd.DataFrame({"a": [1], "b": [2]})
        magic.add("--tracked-variable df_cached")
        del shell.user_ns["df_cached"]
        captured = _capture_lint(monkeypatch, magic)

        magic.lintline("df_cached")

        data = json.loads(captured["body"].decode())["linter_input"]["context"]["data"]
        assert data == "1,2\n"

    def test_needs_valid_symbol_warns(self, magic, monkeypatch):
        capture_display(monkeypatch)
        with capture_warnings() as rec:
            magic.lintline("unknown_var")
        assert_any_warning_contains(rec, MSG_LINTLINE_NEEDS_SYMBOL)

    def test_parse_failure_warns(self, magic, monkeypatch):
        capture_display(monkeypatch)
        with capture_warnings() as rec:
            magic.lintline("--ip badip")
        assert_any_warning_contains(rec, MSG_LINTLINE_PARSE_FAIL)

    def test_connection_error_is_displayed_as_warning(self, magic, shell, monkeypatch):
        capture_display(monkeypatch)
        shell.user_ns["df_err"] = pd.DataFrame({"a": [1]})
        magic.add("--tracked-variable df_err")

        def raise_conn(*a, **k):
            raise LinterConnectionError("timeout")

        monkeypatch.setattr(magic, "http_lint_request", raise_conn)
        with capture_warnings() as rec:
            magic.lintline("df_err")
        assert_any_warning_contains(rec, MSG_CANNOT_REACH)

    def test_http_error_is_displayed_as_warning(self, magic, shell, monkeypatch):
        capture_display(monkeypatch)
        shell.user_ns["df_http"] = pd.DataFrame({"a": [1]})
        magic.add("--tracked-variable df_http")

        def raise_http(*a, **k):
            raise LinterHTTPError(400, "Bad Request", "bad data")

        monkeypatch.setattr(magic, "http_lint_request", raise_http)
        with capture_warnings() as rec:
            magic.lintline("df_http")
        assert_any_warning_contains(rec, MSG_HTTP_REJECT)
        assert_any_warning_contains(rec, "400")


class TestLintCellMagic:
    def test_request_uses_every_tracked_variable_and_then_runs_the_cell(
        self, magic, shell, monkeypatch
    ):
        displayed = capture_display(monkeypatch)
        shell.user_ns["cell_df"] = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        shell.user_ns["cell_arr"] = np.array([5])
        magic.add("--tracked-variable cell_df")
        magic.add("--tracked-variable cell_arr --data-delim | --data-header")
        # Rebind after tracking
        shell.user_ns["cell_df"] = pd.DataFrame({"a": [9], "b": [8]})
        displayed.clear()
        seen = {}

        def recording_fake(ip, port, body, timeout=DEFAULT_TIMEOUT, path="/api/lint"):
            seen["during"] = shell.user_ns.get("cell_flag", "before")
            seen["ip"] = ip
            seen["port"] = port
            seen["body"] = body
            return {"linting_output": "clean"}

        monkeypatch.setattr(magic, "http_lint_request", recording_fake)

        result = magic.lintcell(
            "--show-stats --show-na --show-passing --port 1234",
            "cell_flag = 'after'\ncell_df.iloc[0, 0] = 100\n6 * 7",
        )

        assert seen["during"] == "before"
        assert result == 42
        assert shell.user_ns["cell_flag"] == "after"
        assert shell.user_ns["cell_df"].iloc[0, 0] == 100
        assert seen["port"] == 1234
        assert seen["ip"] == DEFAULT_IP

        payload = json.loads(seen["body"].decode())
        data = payload["linter_input"]["context"]["data"]
        # After refresh, cell_df should be the rebound frame
        assert data["cell_df"][0] == "9,8\n"
        assert data["cell_df"][1] is False
        assert data["cell_df"][2] == ","
        # ndarray with header generates x0
        assert "x0" in data["cell_arr"][0] or "5" in data["cell_arr"][0]
        assert data["cell_arr"][1] is True
        assert data["cell_arr"][2] == "|"
        assert payload["linter_input"]["context"]["code"] == (
            "cell_flag = 'after'\ncell_df.iloc[0, 0] = 100\n6 * 7"
        )
        assert payload["linter_input"]["options"]["show_stats"] is True
        assert_any_display_contains(displayed, "clean")

    def test_deleted_name_keeps_its_cached_csv(self, magic, shell, monkeypatch):
        capture_display(monkeypatch)
        shell.user_ns["keep_df"] = pd.DataFrame({"a": [1]})
        shell.user_ns["drop_df"] = pd.DataFrame({"b": [2]})
        magic.add("--tracked-variable keep_df")
        magic.add("--tracked-variable drop_df --data-header --data-delim ;")
        shell.user_ns["keep_df"] = pd.DataFrame({"a": [9]})
        del shell.user_ns["drop_df"]
        captured = _capture_lint(monkeypatch, magic)

        assert magic.lintcell("", "1") == 1

        data = json.loads(captured["body"].decode())["linter_input"]["context"]["data"]
        assert data["keep_df"] == ["9\n", False, ","]
        assert data["drop_df"] == ["b\n2\n", True, ";"]
        assert json.loads(captured["body"].decode())["linter_input"]["context"]["code"] == "1"

    def test_positional_name_is_ignored_and_empty_tracking_still_posts(
        self, magic, monkeypatch
    ):
        capture_display(monkeypatch)
        captured = _capture_lint(monkeypatch, magic)
        assert magic.lintcell("not_tracked --ip 10.1.1.1 --port 9 --show-passing", "1+1") == 2
        payload = json.loads(captured["body"].decode())
        assert captured["ip"] == "10.1.1.1"
        assert captured["port"] == 9
        assert payload["linter_input"]["context"]["data"] == {}
        assert payload["linter_input"]["context"]["code"] == "1+1"
        assert payload["linter_input"]["options"]["show_passing"] is True

    def test_parse_failure_still_runs_cell(self, magic, shell, monkeypatch):
        capture_display(monkeypatch)
        with capture_warnings() as rec:
            result = magic.lintcell("--ip badip", "3 * 5")
        assert result == 15
        assert_any_warning_contains(rec, MSG_LINTCELL_PARSE_FAIL)

    def test_connection_error_still_runs_cell(self, magic, shell, monkeypatch):
        capture_display(monkeypatch)
        shell.user_ns["err_df"] = pd.DataFrame({"a": [1]})
        magic.add("--tracked-variable err_df")

        def raise_conn(*a, **k):
            raise LinterConnectionError("down")

        monkeypatch.setattr(magic, "http_lint_request", raise_conn)
        with capture_warnings() as rec:
            result = magic.lintcell("", "2 + 2")
        assert result == 4
        assert_any_warning_contains(rec, MSG_CANNOT_REACH)

    def test_other_exception_still_runs_cell(self, magic, shell, monkeypatch):
        capture_display(monkeypatch)
        shell.user_ns["err_df"] = pd.DataFrame({"a": [1]})
        magic.add("--tracked-variable err_df")

        def raise_other(*a, **k):
            raise RuntimeError("unexpected")

        monkeypatch.setattr(magic, "http_lint_request", raise_other)
        with capture_warnings() as rec:
            result = magic.lintcell("", "10")
        assert result == 10
        assert_any_warning_contains(rec, MSG_FAILED_READ_CELL)
