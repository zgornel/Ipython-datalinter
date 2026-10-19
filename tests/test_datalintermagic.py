"""Tests for ``datalintermagic.datalintermagic``.

Maintainability notes:
- Message fragments and payload builders are centralized.
- Display output is captured via a helper (source uses IPython.display.display + Pretty).
- Parsing errors return None + display instead of raising (match current source).
- Only valid identifiers are accepted for tracked variables.
- Cell magic always executes the cell.
"""

import argparse
import json
import socket
import sys
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
    DEFAULT_HEADER,
    DEFAULT_IP,
    DEFAULT_PORT,
    DataLinterMagic,
    LinterConnectionError,
    LinterHTTPError,
    _parse_delimiter,
)

# ---------------------------------------------------------------------------
# Centralized expected message fragments (update here if source wording changes)
# ---------------------------------------------------------------------------
MSG_MISSING_TRACKED = "Warning (Linter): Use '--tracked-variable' to specify a data variable!"
MSG_INVALID_IDENTIFIER = "Warning (Linter): Ivalid identifier specified with '--tracked-variable'"
MSG_LINE_PARSE_ERROR = "Warning (Linter): Line parse error:"
MSG_ADD_PARSE_FAIL = "Warning (Linter): Could not parse line for '%add' magic"
MSG_ADD_FAIL = "Warning (Linter): Could not add"
MSG_LINTLINE_NEEDS_SYMBOL = "Warning (Linter): '%lintline' magic needs a valid data symbol"
MSG_LINTLINE_PARSE_FAIL = "Warning (Linter): Could not parse line for '%lintline' magic"
MSG_LINTCELL_PARSE_FAIL = "Warning (Linter): Could not parse line for '%lintcell' magic"
MSG_CANNOT_REACH = "Warning (Linter): Cannot reach server:"
MSG_HTTP_REJECT = "Warning (Linter): Server rejected the request"
MSG_FAILED_READ = "Warning (Linter): Failed to read linter output"
MSG_LINTER_OUTPUT_PREFIX = "Linter output\n-------------\n"
MSG_GENERATED_NAMES = "will be exported with [x0 ... xN] column names"

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
    return DataLinterMagic(shell)


@pytest.fixture
def two_col_frame():
    return pd.DataFrame({"a": [1, 2], "b": [3, 4]})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def capture_display(monkeypatch):
    """Monkeypatch display() to collect Pretty (or other) objects as strings.

    Returns a list that accumulates the displayed content.
    """
    displayed = []

    def fake_display(obj, *args, **kwargs):
        # Pretty objects expose .data; fall back to str
        text = getattr(obj, "data", None)
        if text is None:
            text = str(obj)
        displayed.append(text)

    monkeypatch.setattr("datalintermagic.datalintermagic.display", fake_display)
    return displayed


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


def _capture_lint(monkeypatch, magic, response=None, error=None):
    """Replace http_lint_request with a recorder that can return a value or raise."""
    captured = {}

    def fake(ip, port, body, timeout=60.0, path="/api/lint"):
        captured["ip"] = ip
        captured["port"] = port
        captured["body"] = body
        captured["timeout"] = timeout
        captured["path"] = path
        captured["calls"] = captured.get("calls", 0) + 1
        if error is not None:
            raise error
        return {"linting_output": "ok"} if response is None else response

    monkeypatch.setattr(magic, "http_lint_request", fake)
    return captured


def make_line_payload(
    data,
    delim=",",
    header=False,
    show_stats=False,
    show_passing=False,
    show_na=False,
    code="",
):
    """Build the JSON body expected by lintline."""
    return {
        "linter_input": {
            "context": {
                "data": data,
                "data_delim": delim,
                "data_header": header,
                "code": code,
            },
            "options": {
                "show_stats": show_stats,
                "show_passing": show_passing,
                "show_na": show_na,
            },
        }
    }


def make_cell_payload(
    tracked,  # dict of name -> list/tuple (csv, header, delim)
    cell,
    show_stats=False,
    show_passing=False,
    show_na=False,
):
    """Build the JSON body expected by lintcell.

    json.dumps converts tuples to lists.
    """
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


def assert_payload_equals(captured_body, expected_dict):
    """Decode and compare a request body against an expected payload dict."""
    actual = json.loads(captured_body.decode("utf-8"))
    assert actual == expected_dict


def assert_any_contains(messages, fragment):
    """Assert that at least one displayed message contains the fragment."""
    assert any(fragment in m for m in messages), f"{fragment!r} not found in {messages}"


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


class TestDefaults:
    def test_constant_values(self):
        assert DEFAULT_IP == "127.0.0.1"
        assert DEFAULT_PORT == 10000
        assert DEFAULT_DELIMITER == ","
        assert DEFAULT_HEADER is False

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
    def test_success_valid_identifier(self, line, expected, monkeypatch):
        # Valid identifiers succeed; no display expected
        displayed = capture_display(monkeypatch)
        assert DataLinterMagic._parse_add_magic(line) == expected
        assert displayed == []

    @pytest.mark.parametrize(
        "line",
        ["", "   ", "--data-header", "--data-delim ,"],
    )
    def test_missing_tracked_variable(self, monkeypatch, line):
        displayed = capture_display(monkeypatch)
        assert DataLinterMagic._parse_add_magic(line) is None
        assert_any_contains(displayed, MSG_MISSING_TRACKED)

    def test_invalid_identifier(self, monkeypatch):
        displayed = capture_display(monkeypatch)
        assert DataLinterMagic._parse_add_magic('--tracked-variable "my var"') is None
        assert_any_contains(displayed, MSG_INVALID_IDENTIFIER)

    def test_unbalanced_quotes_swallowed(self, monkeypatch):
        displayed = capture_display(monkeypatch)
        assert DataLinterMagic._parse_add_magic('--tracked-variable "df') is None
        assert_any_contains(displayed, MSG_LINE_PARSE_ERROR)
        assert_any_contains(displayed, "No closing quotation")


class TestParseLintMagic:
    @pytest.mark.parametrize(
        "line,expected",
        [
            ("", (None, DEFAULT_IP, DEFAULT_PORT, False, False, False)),
            ("   ", (None, DEFAULT_IP, DEFAULT_PORT, False, False, False)),
            ("df", ("df", DEFAULT_IP, DEFAULT_PORT, False, False, False)),
            (
                "--show-stats --show-na --show-passing",
                (None, DEFAULT_IP, DEFAULT_PORT, True, True, True),
            ),
            (
                "--port 65535 --show-na",
                (None, DEFAULT_IP, 65535, False, True, False),
            ),
            ("--port 1", (None, DEFAULT_IP, 1, False, False, False)),
            ("--port +80", (None, DEFAULT_IP, 80, False, False, False)),
            ('"my_var"', ("my_var", DEFAULT_IP, DEFAULT_PORT, False, False, False)),
        ],
    )
    def test_success(self, line, expected, monkeypatch):
        displayed = capture_display(monkeypatch)
        assert DataLinterMagic._parse_lint_magic(line) == expected
        assert displayed == []

    def test_explicit_ipv4(self, monkeypatch):
        displayed = capture_display(monkeypatch)
        parsed = DataLinterMagic._parse_lint_magic(
            "--ip 10.0.0.8 df --port 2500 --show-stats"
        )
        tracked, ip, port, show_stats, show_na, show_passing = parsed
        assert tracked == "df"
        assert ip == "10.0.0.8"
        assert port == 2500
        assert show_stats is True
        assert show_na is False
        assert show_passing is False
        assert displayed == []

    @pytest.mark.parametrize("ip", ["::1", "127.0.0.1"])
    def test_valid_ip(self, ip, monkeypatch):
        displayed = capture_display(monkeypatch)
        parsed_ip = DataLinterMagic._parse_lint_magic(f"--ip {ip}")[1]
        assert parsed_ip == ip
        assert displayed == []

    @pytest.mark.parametrize("ip", ["nope", "256.1.1.1", "127.1"])
    def test_invalid_ip_returns_none_and_displays(self, ip, monkeypatch):
        displayed = capture_display(monkeypatch)
        assert DataLinterMagic._parse_lint_magic(f"--ip {ip}") is None
        assert_any_contains(displayed, MSG_LINE_PARSE_ERROR)
        assert_any_contains(displayed, "not valid")

    @pytest.mark.parametrize(
        "port,fragment",
        [
            ("0", "Invalid port value"),
            ("-1", "Invalid port value"),
            ("65536", "Invalid port value"),
            ("abc", "invalid literal for int"),
            ("1.5", "invalid literal for int"),
            ("", "invalid literal for int"),
        ],
    )
    def test_invalid_port_returns_none_and_displays(self, port, fragment, monkeypatch):
        displayed = capture_display(monkeypatch)
        line = f'--port "{port}"' if port == "" else f"--port {port}"
        assert DataLinterMagic._parse_lint_magic(line) is None
        assert_any_contains(displayed, MSG_LINE_PARSE_ERROR)
        assert_any_contains(displayed, fragment)

    @pytest.mark.parametrize("line", ["--ip", "--port"])
    def test_missing_option_value_returns_none_and_displays(self, line, monkeypatch):
        displayed = capture_display(monkeypatch)
        assert DataLinterMagic._parse_lint_magic(line) is None
        assert_any_contains(displayed, MSG_LINE_PARSE_ERROR)

    @pytest.mark.parametrize("line", ["df extra", "--bogus"])
    def test_unrecognized_args_raise_system_exit(self, line, monkeypatch):
        # argparse calls sys.exit for unrecognized arguments even with exit_on_error=False
        capture_display(monkeypatch)
        assert DataLinterMagic._parse_lint_magic(line) is None

    def test_unbalanced_quotes(self, monkeypatch):
        displayed = capture_display(monkeypatch)
        assert DataLinterMagic._parse_lint_magic('--ip "unterminated') is None
        assert_any_contains(displayed, MSG_LINE_PARSE_ERROR)
        assert_any_contains(displayed, "No closing quotation")

    def test_help_exits(self, monkeypatch):
        # --help still triggers SystemExit inside argparse before our handler
        displayed = capture_display(monkeypatch)
        DataLinterMagic._parse_lint_magic("--help") is None


class TestCsvConversion:
    def test_dataframe_without_header_omits_names_and_index(self, magic, two_col_frame):
        two_col_frame.index = pd.Index(["r1", "r2"], name="row")
        assert magic.dataframe_to_csv_string(two_col_frame, header=False, delimiter=",") == (
            TWO_COL_NO_HEADER
        )

    def test_dataframe_header_uses_string_column_names(self, magic, two_col_frame):
        assert magic.dataframe_to_csv_string(
            two_col_frame, header=True, delimiter=","
        ) == TWO_COL_HEADER
        assert magic.dataframe_to_csv_string(
            two_col_frame, header=True, delimiter="|"
        ) == TWO_COL_PIPE_HEADER

    def test_header_must_be_true_not_merely_truthy(self, magic):
        frame = pd.DataFrame({"a": [1]})
        assert magic.dataframe_to_csv_string(frame, header=None, delimiter=",") == "1\n"
        assert magic.dataframe_to_csv_string(frame, header=1, delimiter=",") == "1\n"
        assert magic.dataframe_to_csv_string(frame, header=True, delimiter=",") == "a\n1\n"

    def test_integer_columns_get_generated_names_and_warn(self, magic, monkeypatch):
        displayed = capture_display(monkeypatch)
        frame = pd.DataFrame([[10, 20], [30, 40]])
        assert magic.dataframe_to_csv_string(frame, True, ",", symbol="arr") == (
            "x0,x1\n10,20\n30,40\n"
        )
        assert_any_contains(displayed, MSG_GENERATED_NAMES)
        assert_any_contains(displayed, "arr")

        # Without header, no warning and no names
        displayed.clear()
        assert magic.dataframe_to_csv_string(frame, False, ",") == "10,20\n30,40\n"
        assert displayed == []

    def test_mixed_columns_trigger_generated_names(self, magic, monkeypatch):
        displayed = capture_display(monkeypatch)
        frame = pd.DataFrame([[1, 2]])
        frame.columns = [0, "b"]  # not all str
        assert magic.dataframe_to_csv_string(frame, True, ",", symbol="mix") == "x0,x1\n1,2\n"
        assert_any_contains(displayed, MSG_GENERATED_NAMES)

    def test_empty_dataframe_is_blank_even_with_header(self, magic):
        empty = pd.DataFrame({"a": pd.Series(dtype="object"), "b": pd.Series(dtype="int")})
        assert empty.empty
        assert magic.dataframe_to_csv_string(empty, header=True, delimiter=";") == "a;b\n"
        assert magic.dataframe_to_csv_string(pd.DataFrame(), False, ",") == ""

    def test_ndarray_two_dimensions(self, magic, monkeypatch):
        displayed = capture_display(monkeypatch)
        arr = np.array([[1, 2], [3, 4]])
        assert magic.ndarray_to_csv_string(arr, False, ",") == "1,2\n3,4\n"
        assert magic.to_csv_string(arr, True, ",", symbol="nd") == "x0,x1\n1,2\n3,4\n"
        assert_any_contains(displayed, MSG_GENERATED_NAMES)

    def test_ndarray_one_dimension(self, magic, monkeypatch):
        displayed = capture_display(monkeypatch)
        arr = np.array([1, 2, 3])
        assert magic.to_csv_string(arr, False, ",") == "1\n2\n3\n"
        assert magic.ndarray_to_csv_string(arr, True, ",", symbol="v") == "x0\n1\n2\n3\n"
        assert_any_contains(displayed, MSG_GENERATED_NAMES)

    def test_empty_ndarray(self, magic):
        assert magic.to_csv_string(np.empty((0, 2)), True, ",") == "x0,x1\n"

    def test_na_rep_is_NA(self, magic):
        frame = pd.DataFrame({"a": [1.0, np.nan]})
        result = magic.dataframe_to_csv_string(frame, header=False, delimiter=",")
        assert "NA" in result
        assert "nan" not in result.lower() or "NA" in result

    @pytest.mark.parametrize(
        "value",
        [[1, 2, 3], pd.Series([1, 2]), "text", 5, {"a": 1}],
    )
    def test_unsupported_types(self, magic, value):
        with pytest.raises(TypeError, match="Unsupported type"):
            magic.to_csv_string(value, False, ",")

    def test_higher_dimensional_ndarray_is_rejected(self, magic):
        with pytest.raises(ValueError):
            magic.to_csv_string(np.ones((2, 2, 2)), False, ",")


class TestHttpLintRequest:
    def test_successful_json_post(self, magic, monkeypatch):
        state = _install_http(monkeypatch)
        state["response"] = state["FakeResponse"](
            200, "OK", '{"linting_output": "caf\u00e9"}'
        )
        body = "é".encode("utf-8")

        parsed = magic.http_lint_request(
            "localhost", "8080", body, timeout=3.5, path="/v2/lint"
        )

        conn = state["conn"]
        assert parsed == {"linting_output": "café"}
        assert conn.host == "localhost"
        assert conn.port == 8080
        assert conn.timeout == 3.5
        assert conn.closed is True
        assert conn.requested["method"] == "POST"
        assert conn.requested["url"] == "/v2/lint"
        assert conn.requested["body"] == body
        assert conn.requested["headers"] == {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Content-Length": str(len(body)),
        }

    def test_default_timeout_and_path(self, magic, monkeypatch):
        state = _install_http(monkeypatch)
        state["response"] = state["FakeResponse"](200, "OK", b'{"ok": true}')

        assert magic.http_lint_request(DEFAULT_IP, DEFAULT_PORT, b"{}") == {"ok": True}
        assert state["conn"].timeout == 60.0
        assert state["conn"].requested["url"] == "/api/lint"
        assert state["conn"].requested["headers"]["Content-Length"] == "2"

    @pytest.mark.parametrize("host", ["0.0.0.0", "::", "[::]"])
    def test_wildcard_hosts_connect_to_loopback(self, magic, monkeypatch, host):
        state = _install_http(monkeypatch)
        state["response"] = state["FakeResponse"](200, "OK", b"{}")

        assert magic.http_lint_request(host, 9, b"{}") == {}
        assert state["conn"].host == DEFAULT_IP
        assert state["conn"].port == 9

    def test_non_200_raises_even_when_body_is_json(self, magic, monkeypatch):
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

    def test_empty_success_body_is_invalid_json(self, magic, monkeypatch):
        state = _install_http(monkeypatch)
        state["response"] = state["FakeResponse"](204, "No Content", b"")
        with pytest.raises(LinterHTTPError) as http_exc:
            magic.http_lint_request("127.0.0.1", 10000, b"{}")
        assert http_exc.value.status == 204
        assert http_exc.value.body == ""

        state["response"] = state["FakeResponse"](200, "OK", b"")
        with pytest.raises(LinterConnectionError, match="Invalid JSON") as exc:
            magic.http_lint_request("127.0.0.1", 10000, b"{}")
        assert isinstance(exc.value.__cause__, json.JSONDecodeError)

    def test_invalid_utf8_success_body(self, magic, monkeypatch):
        state = _install_http(monkeypatch)
        state["response"] = state["FakeResponse"](200, "OK", b"\xff")
        with pytest.raises(LinterConnectionError, match="Invalid JSON"):
            magic.http_lint_request("127.0.0.1", 10000, b"{}")

    def test_invalid_utf8_error_body_is_replaced(self, magic, monkeypatch):
        state = _install_http(monkeypatch)
        state["response"] = state["FakeResponse"](500, "Server Error", b"\xff")
        with pytest.raises(LinterHTTPError) as exc:
            magic.http_lint_request("127.0.0.1", 10000, b"{}")
        assert exc.value.status == 500
        assert exc.value.body == "\ufffd"

    def test_timeout_uses_timeout_message_and_closes(self, magic, monkeypatch):
        state = _install_http(monkeypatch)
        timeout = socket.timeout("slow")
        state["request_error"] = timeout

        with pytest.raises(
            LinterConnectionError, match="Timed out after 2.5s talking to 10.0.0.5:9"
        ) as exc:
            magic.http_lint_request("10.0.0.5", 9, b"{}", timeout=2.5)

        assert exc.value.__cause__ is timeout
        assert state["conn"].closed is True

    @pytest.mark.parametrize(
        "error",
        [
            ConnectionRefusedError("refused"),
            socket.gaierror("name lookup failed"),
            OSError("network down"),
        ],
    )
    def test_unreachable_server(self, magic, monkeypatch, error):
        state = _install_http(monkeypatch)
        state["request_error"] = error

        with pytest.raises(LinterConnectionError, match="Is the Docker server running") as exc:
            magic.http_lint_request("10.0.0.8", 4242, b"{}")

        message = str(exc.value)
        assert "Could not reach DataLinter at 10.0.0.8:4242" in message
        assert exc.value.__cause__ is error
        assert state["conn"].closed is True

    def test_oserror_while_reading_the_response(self, magic, monkeypatch):
        state = _install_http(monkeypatch)
        state["response"] = state["FakeResponse"](200, "OK", b"{}")
        reset = OSError("connection reset")
        state["response_error"] = reset

        with pytest.raises(LinterConnectionError, match="Could not reach DataLinter") as exc:
            magic.http_lint_request("127.0.0.1", 10000, b"{}")
        assert exc.value.__cause__ is reset
        assert state["conn"].closed is True

    def test_wildcard_host_is_named_in_the_timeout(self, magic, monkeypatch):
        state = _install_http(monkeypatch)
        state["request_error"] = socket.timeout("slow")
        with pytest.raises(LinterConnectionError, match=r"talking to 127\.0\.0\.1:9"):
            magic.http_lint_request("::", 9, b"{}", timeout=1)


class TestAddMagic:
    def test_tracks_dataframe_and_returns_none(self, magic, shell, monkeypatch, two_col_frame):
        displayed = capture_display(monkeypatch)
        shell.user_ns["df_add"] = two_col_frame

        assert magic.add("--tracked-variable df_add") is None
        assert displayed == []

        csv_text, header, delim = magic.all_tracked_variables["df_add"]
        assert (csv_text, header, delim) == (TWO_COL_NO_HEADER, False, ",")

    def test_tracks_dataframe_with_header_and_custom_delim(self, magic, shell, two_col_frame, monkeypatch):
        capture_display(monkeypatch)  # silence any display
        shell.user_ns["df_add"] = two_col_frame
        magic.add("--tracked-variable df_add --data-header --data-delim |")
        assert magic.all_tracked_variables["df_add"] == (
            TWO_COL_PIPE_HEADER,
            True,
            "|",
        )

    def test_tracks_ndarray(self, magic, shell, monkeypatch):
        displayed = capture_display(monkeypatch)
        shell.user_ns["arr_add"] = np.array([[1, 2], [3, 4]])
        magic.add("--tracked-variable arr_add --data-header")
        assert magic.all_tracked_variables["arr_add"] == (
            "x0,x1\n1,2\n3,4\n",
            True,
            ",",
        )
        # Generated names produce a warning
        assert_any_contains(displayed, MSG_GENERATED_NAMES)

    def test_rejects_expression_and_spaced_name(self, magic, shell, monkeypatch):
        displayed = capture_display(monkeypatch)
        shell.user_ns["df_expr"] = pd.DataFrame({"a": [1, 2]})
        # Not a valid identifier
        assert magic.add('--tracked-variable "df_expr * 2"') is None
        assert "df_expr * 2" not in magic.all_tracked_variables
        assert_any_contains(displayed, MSG_INVALID_IDENTIFIER)
        assert_any_contains(displayed, MSG_ADD_PARSE_FAIL)

        displayed.clear()
        shell.user_ns["my_frame"] = pd.DataFrame({"a": [8]})
        # Valid identifier works
        assert magic.add("--tracked-variable my_frame") is None
        assert "my_frame" in magic.all_tracked_variables

    def test_missing_name_warns_and_does_not_track(self, magic, monkeypatch):
        displayed = capture_display(monkeypatch)
        assert magic.add("--tracked-variable does_not_exist") is None
        assert "does_not_exist" not in magic.all_tracked_variables
        assert_any_contains(displayed, MSG_ADD_FAIL)

    def test_unsupported_value_warns_and_does_not_track(self, magic, shell, monkeypatch):
        displayed = capture_display(monkeypatch)
        shell.user_ns["vals_add"] = [1, 2, 3]
        assert magic.add("--tracked-variable vals_add") is None
        assert "vals_add" not in magic.all_tracked_variables
        assert_any_contains(displayed, MSG_ADD_FAIL)
        assert_any_contains(displayed, "Unsupported type")

    def test_bad_syntax_warns_instead_of_raising(self, magic, monkeypatch):
        displayed = capture_display(monkeypatch)
        assert magic.add('--tracked-variable "df') is None
        assert magic.all_tracked_variables == {}
        assert_any_contains(displayed, MSG_LINE_PARSE_ERROR)
        assert_any_contains(displayed, MSG_ADD_PARSE_FAIL)

    def test_unknown_argument_raises_system_exit(self, magic, monkeypatch):
        capture_display(monkeypatch)
        assert magic.add("--unknown") is None
        assert magic.all_tracked_variables == {}

    def test_shell_line_magic_uses_the_registered_instance(self, shell, monkeypatch):
        capture_display(monkeypatch)
        shell.register_magics(DataLinterMagic)
        registered = shell.find_line_magic("add").__self__
        registered.all_tracked_variables.clear()
        shell.user_ns["reg_df"] = pd.DataFrame({"a": [4]})

        assert shell.run_line_magic("add", "--tracked-variable reg_df") is None
        assert registered.all_tracked_variables["reg_df"] == ("4\n", False, ",")
        assert shell.find_cell_magic("lintcell").__self__ is registered


class TestLintLineMagic:
    def test_posts_tracked_frame_and_displays_output(self, magic, shell, monkeypatch):
        displayed = capture_display(monkeypatch)
        shell.user_ns["df_line"] = pd.DataFrame({"city": ["Żory", "Łódź"]})
        magic.add("--tracked-variable df_line --data-header --data-delim ;")
        displayed.clear()
        captured = _capture_lint(
            monkeypatch, magic, response={"linting_output": "row 1: ok\nrow 2: ok"}
        )

        assert magic.lintline("df_line --show-stats --port 2222") is None

        assert captured["ip"] == DEFAULT_IP
        assert captured["port"] == 2222
        assert captured["path"] == "/api/lint"
        assert_payload_equals(
            captured["body"],
            make_line_payload(
                data="city\nŻory\nŁódź\n",
                delim=";",
                header=True,
                show_stats=True,
            ),
        )
        assert_any_contains(displayed, MSG_LINTER_OUTPUT_PREFIX + "row 1: ok\nrow 2: ok")

    def test_refreshes_from_user_namespace_before_the_request(self, magic, shell, monkeypatch):
        capture_display(monkeypatch)
        frame = pd.DataFrame({"a": [1], "b": [2]})
        shell.user_ns["df_refresh"] = frame
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

    def test_keeps_cached_csv_when_the_name_was_deleted(self, magic, shell, monkeypatch):
        capture_display(monkeypatch)
        shell.user_ns["df_cached"] = pd.DataFrame({"a": [1], "b": [2]})
        magic.add("--tracked-variable df_cached")
        del shell.user_ns["df_cached"]
        captured = _capture_lint(monkeypatch, magic)

        magic.lintline("df_cached")

        assert json.loads(captured["body"].decode())["linter_input"]["context"]["data"] == "1,2\n"

    def test_explicit_ip_is_forwarded(self, magic, shell, monkeypatch):
        capture_display(monkeypatch)
        shell.user_ns["df_ip"] = pd.DataFrame({"a": [1]})
        magic.add("--tracked-variable df_ip")
        captured = _capture_lint(monkeypatch, magic)

        magic.lintline("df_ip --ip 10.1.1.1 --port 9")

        assert captured["ip"] == "10.1.1.1"
        assert captured["port"] == 9

    def test_unknown_variable_displays_warning(self, magic, monkeypatch):
        displayed = capture_display(monkeypatch)
        captured = _capture_lint(monkeypatch, magic)
        assert magic.lintline("not_tracked") is None
        assert "calls" not in captured
        assert_any_contains(displayed, MSG_LINTLINE_NEEDS_SYMBOL)

    def test_empty_line_displays_needs_symbol(self, magic, monkeypatch):
        displayed = capture_display(monkeypatch)
        captured = _capture_lint(monkeypatch, magic)
        assert magic.lintline("") is None
        assert "calls" not in captured
        assert_any_contains(displayed, MSG_LINTLINE_NEEDS_SYMBOL)

    def test_connection_error_is_displayed(self, magic, shell, monkeypatch):
        displayed = capture_display(monkeypatch)
        shell.user_ns["df_down"] = pd.DataFrame({"a": [1]})
        magic.add("--tracked-variable df_down")
        shell.user_ns["df_down"] = pd.DataFrame({"a": [1, 2]})
        _capture_lint(monkeypatch, magic, error=LinterConnectionError("nope"))

        assert magic.lintline("df_down") is None

        assert_any_contains(displayed, MSG_CANNOT_REACH)
        # The namespace was refreshed before the failing request.
        assert magic.all_tracked_variables["df_down"][0] == "1\n2\n"

    def test_http_error_is_displayed(self, magic, shell, monkeypatch):
        displayed = capture_display(monkeypatch)
        shell.user_ns["df_http"] = pd.DataFrame({"a": [1]})
        magic.add("--tracked-variable df_http")
        _capture_lint(
            monkeypatch,
            magic,
            error=LinterHTTPError(422, "Unprocessable Entity", "bad column"),
        )

        magic.lintline("df_http")

        assert_any_contains(displayed, MSG_HTTP_REJECT)
        assert_any_contains(displayed, "HTTP 422 Unprocessable Entity")
        assert_any_contains(displayed, "bad column")

    def test_missing_linting_output_is_displayed_as_failure(self, magic, shell, monkeypatch):
        displayed = capture_display(monkeypatch)
        shell.user_ns["df_bad_json"] = pd.DataFrame({"a": [1]})
        magic.add("--tracked-variable df_bad_json")
        _capture_lint(monkeypatch, magic, response={"nope": 1})

        magic.lintline("df_bad_json")

        assert_any_contains(displayed, MSG_FAILED_READ)

    def test_generic_request_error_is_displayed(self, magic, shell, monkeypatch):
        displayed = capture_display(monkeypatch)
        shell.user_ns["df_boom"] = pd.DataFrame({"a": [1]})
        magic.add("--tracked-variable df_boom")
        _capture_lint(monkeypatch, magic, error=RuntimeError("boom"))

        magic.lintline("df_boom")
        assert_any_contains(displayed, "boom")
        assert_any_contains(displayed, MSG_FAILED_READ)

    def test_refresh_type_error_propagates_and_skips_the_request(self, magic, shell, monkeypatch):
        capture_display(monkeypatch)
        shell.user_ns["df_type"] = pd.DataFrame({"a": [1]})
        magic.add("--tracked-variable df_type")
        original = magic.all_tracked_variables["df_type"]
        shell.user_ns["df_type"] = (1, 2, 3)
        captured = _capture_lint(monkeypatch, magic)

        with pytest.raises(TypeError, match="Unsupported type"):
            magic.lintline("df_type")

        assert "calls" not in captured
        assert magic.all_tracked_variables["df_type"] == original

    def test_invalid_port_displays_and_skips_request(self, magic, monkeypatch):
        displayed = capture_display(monkeypatch)
        captured = _capture_lint(monkeypatch, magic)
        assert magic.lintline("df_line --port 0") is None
        assert "calls" not in captured
        assert_any_contains(displayed, MSG_LINE_PARSE_ERROR)
        assert_any_contains(displayed, MSG_LINTLINE_PARSE_FAIL)


class TestLintCellMagic:
    def test_request_uses_every_tracked_variable_and_then_runs_the_cell(
        self, magic, shell, monkeypatch
    ):
        displayed = capture_display(monkeypatch)
        shell.user_ns["cell_df"] = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        shell.user_ns["cell_arr"] = np.array([5])
        magic.add("--tracked-variable cell_df")
        magic.add("--tracked-variable cell_arr --data-delim | --data-header")
        shell.user_ns["cell_df"] = pd.DataFrame({"a": [9], "b": [8]})
        displayed.clear()
        seen = {}

        def recording_fake(ip, port, body, timeout=60.0, path="/api/lint"):
            seen["during"] = shell.user_ns["cell_flag"]
            seen["ip"] = ip
            seen["port"] = port
            seen["body"] = body
            return {"linting_output": "clean"}

        monkeypatch.setattr(magic, "http_lint_request", recording_fake)
        shell.user_ns["cell_flag"] = "before"

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
        assert_payload_equals(
            seen["body"],
            make_cell_payload(
                tracked={
                    "cell_df": ["9,8\n", False, ","],
                    "cell_arr": ["x0\n5\n", True, "|"],
                },
                cell="cell_flag = 'after'\ncell_df.iloc[0, 0] = 100\n6 * 7",
                show_stats=True,
                show_passing=True,
                show_na=True,
            ),
        )
        assert_any_contains(displayed, MSG_LINTER_OUTPUT_PREFIX + "clean")

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
        assert payload["linter_input"]["options"] == {
            "show_stats": False,
            "show_passing": True,
            "show_na": False,
        }

    def test_server_errors_do_not_block_the_cell(self, magic, shell, monkeypatch):
        displayed = capture_display(monkeypatch)
        shell.user_ns["ran_conn"] = False
        _capture_lint(monkeypatch, magic, error=LinterConnectionError("down"))
        assert magic.lintcell("", "ran_conn = True\n2 + 2") == 4
        assert shell.user_ns["ran_conn"] is True
        assert_any_contains(displayed, MSG_CANNOT_REACH)

        displayed.clear()
        shell.user_ns["ran_http"] = False
        _capture_lint(
            monkeypatch,
            magic,
            error=LinterHTTPError(500, "Internal Server Error", "nope"),
        )
        assert magic.lintcell("", "ran_http = True\n3") == 3
        assert shell.user_ns["ran_http"] is True
        assert_any_contains(displayed, "HTTP 500 Internal Server Error")

        displayed.clear()
        _capture_lint(monkeypatch, magic, response={})
        assert magic.lintcell("", "4") == 4
        assert_any_contains(displayed, MSG_FAILED_READ)

    def test_parse_errors_display_but_still_run_the_cell(self, magic, shell, monkeypatch):
        displayed = capture_display(monkeypatch)
        shell.user_ns["cell_guard"] = 0
        captured = _capture_lint(monkeypatch, magic)

        # Invalid IP is caught (ValueError), returns None, but cell still runs.
        # Pure assignment returns None from run_cell; check the side effect.
        assert magic.lintcell("--ip nope", "cell_guard = 1") is None
        assert shell.user_ns["cell_guard"] == 1
        assert_any_contains(displayed, MSG_LINTCELL_PARSE_FAIL)
        assert "calls" not in captured

        displayed.clear()
        shell.user_ns["cell_guard"] = 0
        assert magic.lintcell("--port 99999", "cell_guard = 2") is None
        assert shell.user_ns["cell_guard"] == 2
        assert_any_contains(displayed, MSG_LINTCELL_PARSE_FAIL)

        shell.user_ns["cell_guard"] = 0
        assert magic.lintcell("extra junk", "cell_guard = 3") is None
        assert shell.user_ns["cell_guard"] == 3

    def test_refresh_failure_propagates_before_cell(self, magic, shell, monkeypatch):
        capture_display(monkeypatch)
        shell.user_ns["cell_bad"] = pd.DataFrame({"a": [1]})
        magic.add("--tracked-variable cell_bad")
        shell.user_ns["cell_bad"] = {"a": 1}
        shell.user_ns["cell_bad_ran"] = False
        captured = _capture_lint(monkeypatch, magic)

        with pytest.raises(TypeError, match="Unsupported type"):
            magic.lintcell("", "cell_bad_ran = True")

        assert shell.user_ns["cell_bad_ran"] is False
        assert "calls" not in captured

    def test_registered_cell_magic_returns_the_cell_result(self, shell, monkeypatch):
        capture_display(monkeypatch)
        shell.register_magics(DataLinterMagic)
        registered = shell.find_cell_magic("lintcell").__self__
        registered.all_tracked_variables.clear()
        _capture_lint(monkeypatch, registered, response={"linting_output": "via shell"})

        assert shell.run_cell_magic("lintcell", "--show-stats", "10 - 3") == 7
