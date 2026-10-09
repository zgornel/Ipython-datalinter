"""Tests for ``datalintermagic.datalintermagic``.
"""

import argparse
import json
import socket
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

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

TWO_COL_NO_HEADER = "1,3\n2,4\n"
TWO_COL_HEADER = "a,b\n1,3\n2,4\n"
TWO_COL_PIPE_HEADER = "a|b\n1|3\n2|4\n"


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


def _install_http(monkeypatch):
    """Replace HTTPConnection with a fake that records one request."""
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
    captured = {}

    def fake(ip, port, body, timeout=300.0, path="/api/lint"):
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


def _two_col_frame():
    return pd.DataFrame({"a": [1, 2], "b": [3, 4]})


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
            ('--tracked-variable "my var"', ("my var", False, ",")),
            ("--tracked-variable df --data-delim ''", ("df", False, "")),
        ],
    )
    def test_success(self, line, expected):
        assert DataLinterMagic._parse_add_magic(line) == expected

    @pytest.mark.parametrize(
        "line",
        ["", "   ", "--data-header", "--data-delim ,"],
    )
    def test_missing_tracked_variable_warns_and_returns_none(self, capsys, line):
        assert DataLinterMagic._parse_add_magic(line) is None
        assert (
            "Warning (Linter): Use '--tracked-variable' to specify a data variable!"
            in capsys.readouterr().out
        )

    def test_unbalanced_quotes_are_swallowed(self, capsys):
        assert DataLinterMagic._parse_add_magic('--tracked-variable "df') is None
        out = capsys.readouterr().out
        assert "Could not parse line for %%add line magic:" in out
        assert "No closing quotation" in out

    @pytest.mark.parametrize("line", ["--nope", "--tracked-variable", "--data-delim"])
    def test_argparse_failures_exit(self, line):
        # The add parser keeps argparse's default exit_on_error=True.
        with pytest.raises(SystemExit) as exc:
            DataLinterMagic._parse_add_magic(line)
        assert exc.value.code == 2


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
            ('"my var"', ("my var", DEFAULT_IP, DEFAULT_PORT, False, False, False)),
        ],
    )
    def test_success_without_explicit_ip(self, line, expected):
        assert DataLinterMagic._parse_lint_magic(line) == expected

    def test_explicit_ipv4_is_returned_as_packed_bytes(self):
        # inet_aton validates the address and its bytes object is what the
        # parser returns (not the original string).
        parsed = DataLinterMagic._parse_lint_magic(
            "--ip 10.0.0.8 df --port 2500 --show-stats"
        )
        tracked, ip, port, show_stats, show_na, show_passing = parsed
        assert tracked == "df"
        assert ip == socket.inet_aton("10.0.0.8")
        assert port == 2500
        assert show_stats is True
        assert show_na is False
        assert show_passing is False

    def test_abbreviated_ipv4_is_accepted(self):
        ip = DataLinterMagic._parse_lint_magic("--ip 127.1")[1]
        assert ip == socket.inet_aton("127.0.0.1")

    @pytest.mark.parametrize("ip", ["nope", "::1", "256.1.1.1", ""])
    def test_invalid_ip_raises(self, ip):
        with pytest.raises(ValueError, match="not valid") as exc:
            DataLinterMagic._parse_lint_magic(f"--ip {ip}" if ip else '--ip ""')
        assert isinstance(exc.value.__cause__, OSError)

    @pytest.mark.parametrize(
        "port,match",
        [
            ("0", "Invalid port value. Needs to be an integer between 1 and 65535"),
            ("-1", "Invalid port value. Needs to be an integer between 1 and 65535"),
            ("65536", "Invalid port value. Needs to be an integer between 1 and 65535"),
            ("abc", "invalid literal for int"),
            ("1.5", "invalid literal for int"),
            ("", "invalid literal for int"),
        ],
    )
    def test_invalid_port_raises(self, port, match):
        line = f'--port "{port}"' if port == "" else f"--port {port}"
        with pytest.raises(ValueError, match=match):
            DataLinterMagic._parse_lint_magic(line)

    @pytest.mark.parametrize("line", ["--ip", "--port"])
    def test_missing_option_value_raises_argument_error(self, line):
        # exit_on_error=False lets ArgumentError escape; it is not wrapped.
        with pytest.raises(argparse.ArgumentError):
            DataLinterMagic._parse_lint_magic(line)

    def test_unbalanced_quotes_propagate(self):
        with pytest.raises(ValueError, match="No closing quotation"):
            DataLinterMagic._parse_lint_magic('--ip "unterminated')

    @pytest.mark.parametrize(
        "line,code",
        [
            ("--help", 0),
        ],
    )
    def test_parser_exits_for_usage_errors(self, line, code):
        # parse_args() still calls sys.exit for --help.
        with pytest.raises(SystemExit) as exc:
            DataLinterMagic._parse_lint_magic(line)
        assert exc.value.code == code
    
    @pytest.mark.parametrize(
        "line,code",
        [
            ("df extra", 2),
            ("--bogus", 2),
        ],
    )
    def test_parser_exits_for_usage_errors_2(self, line, code):
        with pytest.raises(argparse.ArgumentError) as exc:
            DataLinterMagic._parse_lint_magic(line)

class TestCsvConversion:
    def test_dataframe_without_header_omits_names_and_index(self, magic):
        frame = _two_col_frame()
        frame.index = pd.Index(["r1", "r2"], name="row")
        assert magic.dataframe_to_csv_string(frame, header=False, delimiter=",") == (
            TWO_COL_NO_HEADER
        )

    def test_dataframe_header_uses_string_column_names(self, magic):
        assert magic.dataframe_to_csv_string(
            _two_col_frame(), header=True, delimiter=","
        ) == TWO_COL_HEADER
        assert magic.dataframe_to_csv_string(
            _two_col_frame(), header=True, delimiter="|"
        ) == TWO_COL_PIPE_HEADER

    def test_header_must_be_true_not_merely_truthy(self, magic):
        frame = pd.DataFrame({"a": [1]})
        assert magic.dataframe_to_csv_string(frame, header=None, delimiter=",") == "1\n"
        assert magic.dataframe_to_csv_string(frame, header=1, delimiter=",") == "1\n"
        assert magic.dataframe_to_csv_string(frame, header=True, delimiter=",") == "a\n1\n"

    def test_integer_columns_get_generated_names_when_header_requested(self, magic):
        frame = pd.DataFrame([[10, 20], [30, 40]])
        assert magic.dataframe_to_csv_string(frame, True, ",") == "x0,x1\n10,20\n30,40\n"
        assert magic.dataframe_to_csv_string(frame, False, ",") == "10,20\n30,40\n"

    def test_generated_names_depend_only_on_the_first_column(self, magic):
        frame = pd.DataFrame([[1, 2]])
        frame.columns = [0, "b"]
        assert magic.dataframe_to_csv_string(frame, True, ",") == "x0,x1\n1,2\n"

        named = pd.DataFrame([[1, 2]])
        named.columns = ["a", 1]
        assert magic.dataframe_to_csv_string(named, True, ",") == "a,1\n1,2\n"

    def test_empty_dataframe_is_blank_even_with_header(self, magic):
        empty = pd.DataFrame({"a": pd.Series(dtype="object"), "b": pd.Series(dtype="int")})
        assert empty.empty
        assert magic.dataframe_to_csv_string(empty, header=True, delimiter=";") == ""
        assert magic.dataframe_to_csv_string(pd.DataFrame(), False, ",") == ""

    def test_ndarray_two_dimensions(self, magic):
        arr = np.array([[1, 2], [3, 4]])
        assert magic.ndarray_to_csv_string(arr, False, ",") == "1,2\n3,4\n"
        assert magic.to_csv_string(arr, True, ",") == "x0,x1\n1,2\n3,4\n"

    def test_ndarray_one_dimension(self, magic):
        arr = np.array([1, 2, 3])
        assert magic.to_csv_string(arr, False, ",") == "1\n2\n3\n"
        assert magic.ndarray_to_csv_string(arr, True, ",") == "x0\n1\n2\n3\n"

    def test_empty_ndarray(self, magic):
        assert magic.to_csv_string(np.empty((0, 2)), True, ",") == ""

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
        assert state["conn"].timeout == 300.0
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

        with pytest.raises(LinterConnectionError, match="Timed out after 2.5s talking to 10.0.0.5:9") as exc:
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
    def test_tracks_dataframe_and_returns_none(self, magic, shell, capsys):
        shell.user_ns["df_add"] = _two_col_frame()

        assert magic.add("--tracked-variable df_add") is None
        assert capsys.readouterr().out == ""

        csv_text, header, delim = magic.all_tracked_variables["df_add"]
        assert (csv_text, header, delim) == (TWO_COL_NO_HEADER, False, ",")

        magic.add("--tracked-variable df_add --data-header --data-delim |")
        assert magic.all_tracked_variables["df_add"] == (
            TWO_COL_PIPE_HEADER,
            True,
            "|",
        )

    def test_tracks_ndarray(self, magic, shell):
        shell.user_ns["arr_add"] = np.array([[1, 2], [3, 4]])
        magic.add("--tracked-variable arr_add --data-header")
        assert magic.all_tracked_variables["arr_add"] == (
            "x0,x1\n1,2\n3,4\n",
            True,
            ",",
        )

    def test_evaluates_expressions(self, magic, shell):
        shell.user_ns["df_expr"] = pd.DataFrame({"a": [1, 2]})
        assert magic.add('--tracked-variable "df_expr * 2"') is None
        assert magic.all_tracked_variables["df_expr * 2"] == ("2\n4\n", False, ",")

    def test_non_expression_name_is_not_tracked(self, magic, shell, capsys):
        # shlex keeps the spaced name, but add() evaluates it with shell.ev.
        shell.user_ns["my frame"] = pd.DataFrame({"a": [8]})
        assert magic.add('--tracked-variable "my frame"') is None
        assert "my frame" not in magic.all_tracked_variables
        out = capsys.readouterr().out
        assert "Could not add 'my frame'" in out
        assert "invalid syntax" in out

    def test_missing_name_warns_and_does_not_track(self, magic, capsys):
        assert magic.add("--tracked-variable does_not_exist") is None
        assert "does_not_exist" not in magic.all_tracked_variables
        out = capsys.readouterr().out
        assert "Could not add 'does_not_exist'" in out

    def test_unsupported_value_warns_and_does_not_track(self, magic, shell, capsys):
        shell.user_ns["vals_add"] = [1, 2, 3]
        assert magic.add("--tracked-variable vals_add") is None
        assert "vals_add" not in magic.all_tracked_variables
        out = capsys.readouterr().out
        assert "Could not add 'vals_add'" in out
        assert "Unsupported type" in out

    def test_bad_syntax_warns_instead_of_raising(self, magic, capsys):
        assert magic.add('--tracked-variable "df') is None
        assert magic.all_tracked_variables == {}
        assert "Could not parse line for %%add line magic:" in capsys.readouterr().out

    def test_unknown_argument_exits_without_tracking(self, magic):
        with pytest.raises(SystemExit):
            magic.add("--bogus")
        assert magic.all_tracked_variables == {}

    def test_shell_line_magic_uses_the_registered_instance(self, shell):
        shell.register_magics(DataLinterMagic)
        registered = shell.find_line_magic("add").__self__
        registered.all_tracked_variables.clear()
        shell.user_ns["reg_df"] = pd.DataFrame({"a": [4]})

        assert shell.run_line_magic("add", "--tracked-variable reg_df") is None
        assert registered.all_tracked_variables["reg_df"] == ("4\n", False, ",")
        assert shell.find_cell_magic("lintcell").__self__ is registered


class TestLintLineMagic:
    def test_posts_tracked_frame_and_prints_output(self, magic, shell, monkeypatch, capsys):
        shell.user_ns["df_line"] = pd.DataFrame({"city": ["Żory", "Łódź"]})
        magic.add("--tracked-variable df_line --data-header --data-delim ;")
        capsys.readouterr()
        captured = _capture_lint(
            monkeypatch, magic, response={"linting_output": "row 1: ok\nrow 2: ok"}
        )

        assert magic.lintline("df_line --show-stats --port 2222") is None

        assert captured["ip"] == DEFAULT_IP
        assert captured["port"] == 2222
        assert captured["path"] == "/api/lint"
        assert json.loads(captured["body"].decode("utf-8")) == {
            "linter_input": {
                "context": {
                    "data": "city\nŻory\nŁódź\n",
                    "data_delim": ";",
                    "data_header": True,
                    "code": "",
                },
                "options": {
                    "show_stats": True,
                    "show_passing": False,
                    "show_na": False,
                },
            }
        }
        assert capsys.readouterr().out == (
            "Linter output\n-------------\nrow 1: ok\nrow 2: ok\n"
        )

    def test_refreshes_from_user_namespace_before_the_request(self, magic, shell, monkeypatch):
        frame = pd.DataFrame({"a": [1], "b": [2]})
        shell.user_ns["df_refresh"] = frame
        magic.add("--tracked-variable df_refresh")
        frame.iloc[0, 0] = 7
        shell.user_ns["df_refresh"] = pd.DataFrame({"a": [7, 8], "b": [2, 9]})
        captured = _capture_lint(monkeypatch, magic)

        magic.lintline("df_refresh --show-na --show-passing")

        assert json.loads(captured["body"].decode()) == {
            "linter_input": {
                "context": {
                    "data": "7,2\n8,9\n",
                    "data_delim": ",",
                    "data_header": False,
                    "code": "",
                },
                "options": {
                    "show_stats": False,
                    "show_passing": True,
                    "show_na": True,
                },
            }
        }
        assert magic.all_tracked_variables["df_refresh"][0] == "7,2\n8,9\n"

    def test_keeps_cached_csv_when_the_name_was_deleted(self, magic, shell, monkeypatch):
        shell.user_ns["df_cached"] = pd.DataFrame({"a": [1], "b": [2]})
        magic.add("--tracked-variable df_cached")
        del shell.user_ns["df_cached"]
        captured = _capture_lint(monkeypatch, magic)

        magic.lintline("df_cached")

        assert json.loads(captured["body"].decode())["linter_input"]["context"]["data"] == "1,2\n"

    def test_explicit_ip_is_forwarded_as_packed_bytes(self, magic, shell, monkeypatch):
        shell.user_ns["df_ip"] = pd.DataFrame({"a": [1]})
        magic.add("--tracked-variable df_ip")
        captured = _capture_lint(monkeypatch, magic)

        magic.lintline("df_ip --ip 10.1.1.1 --port 9")

        assert captured["ip"] == socket.inet_aton("10.1.1.1")
        assert captured["port"] == 9

    def test_unknown_variable_is_a_silent_no_op(self, magic, monkeypatch, capsys):
        captured = _capture_lint(monkeypatch, magic)
        assert magic.lintline("not_tracked") is None
        assert magic.lintline("") is None
        assert "calls" not in captured
        assert capsys.readouterr().out == ""

    def test_connection_error_is_printed(self, magic, shell, monkeypatch, capsys):
        shell.user_ns["df_down"] = pd.DataFrame({"a": [1]})
        magic.add("--tracked-variable df_down")
        shell.user_ns["df_down"] = pd.DataFrame({"a": [1, 2]})
        _capture_lint(monkeypatch, magic, error=LinterConnectionError("nope"))

        assert magic.lintline("df_down") is None

        assert "Warning (Linter): Cannot reach server:" in capsys.readouterr().out
        # The namespace was refreshed before the failing request.
        assert magic.all_tracked_variables["df_down"][0] == "1\n2\n"

    def test_http_error_is_printed(self, magic, shell, monkeypatch, capsys):
        shell.user_ns["df_http"] = pd.DataFrame({"a": [1]})
        magic.add("--tracked-variable df_http")
        _capture_lint(
            monkeypatch,
            magic,
            error=LinterHTTPError(422, "Unprocessable Entity", "bad column"),
        )

        magic.lintline("df_http")

        out = capsys.readouterr().out
        assert "Server rejected the request (HTTP 422 Unprocessable Entity):" in out
        assert "bad column" in out

    def test_missing_linting_output_is_printed_as_failure(self, magic, shell, monkeypatch, capsys):
        shell.user_ns["df_bad_json"] = pd.DataFrame({"a": [1]})
        magic.add("--tracked-variable df_bad_json")
        _capture_lint(monkeypatch, magic, response={"nope": 1})

        magic.lintline("df_bad_json")

        out = capsys.readouterr().out
        assert "Failed to read linter output (perhaps linting failed):" in out

    def test_generic_request_error_is_printed(self, magic, shell, monkeypatch, capsys):
        shell.user_ns["df_boom"] = pd.DataFrame({"a": [1]})
        magic.add("--tracked-variable df_boom")
        _capture_lint(monkeypatch, magic, error=RuntimeError("boom"))

        magic.lintline("df_boom")
        out = capsys.readouterr().out
        assert "boom" in out
        assert "Failed to read linter output" in out

    def test_refresh_type_error_propagates_and_skips_the_request(self, magic, shell, monkeypatch):
        shell.user_ns["df_type"] = pd.DataFrame({"a": [1]})
        magic.add("--tracked-variable df_type")
        original = magic.all_tracked_variables["df_type"]
        shell.user_ns["df_type"] = (1, 2, 3)
        captured = _capture_lint(monkeypatch, magic)

        with pytest.raises(TypeError, match="Unsupported type"):
            magic.lintline("df_type")

        assert "calls" not in captured
        assert magic.all_tracked_variables["df_type"] == original

    def test_invalid_port_raises_before_the_request(self, magic, monkeypatch):
        captured = _capture_lint(monkeypatch, magic)
        with pytest.raises(ValueError, match="Invalid port"):
            magic.lintline("df_line --port 0")
        assert "calls" not in captured


class TestLintCellMagic:
    def test_request_uses_every_tracked_variable_and_then_runs_the_cell(
        self, magic, shell, monkeypatch, capsys
    ):
        shell.user_ns["cell_df"] = _two_col_frame()
        shell.user_ns["cell_arr"] = np.array([5])
        magic.add("--tracked-variable cell_df")
        magic.add("--tracked-variable cell_arr --data-delim | --data-header")
        shell.user_ns["cell_df"] = pd.DataFrame({"a": [9], "b": [8]})
        seen = {}

        def recording_fake(ip, port, body, timeout=300.0, path="/api/lint"):
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
        assert json.loads(seen["body"].decode()) == {
            "linter_input": {
                "context": {
                    "data": {
                        "cell_df": ["9,8\n", False, ","],
                        "cell_arr": ["x0\n5\n", True, "|"],
                    },
                    "code": "cell_flag = 'after'\ncell_df.iloc[0, 0] = 100\n6 * 7",
                },
                "options": {
                    "show_stats": True,
                    "show_passing": True,
                    "show_na": True,
                },
            }
        }
        assert capsys.readouterr().out.startswith(
            "Linter output\n-------------\nclean\n"
        )

    def test_deleted_name_keeps_its_cached_csv(self, magic, shell, monkeypatch):
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
        captured = _capture_lint(monkeypatch, magic)
        assert magic.lintcell("not_tracked --ip 10.1.1.1 --port 9 --show-passing", "1+1") == 2
        payload = json.loads(captured["body"].decode())
        assert captured["ip"] == socket.inet_aton("10.1.1.1")
        assert captured["port"] == 9
        assert payload["linter_input"]["context"]["data"] == {}
        assert payload["linter_input"]["context"]["code"] == "1+1"
        assert payload["linter_input"]["options"] == {
            "show_stats": False,
            "show_passing": True,
            "show_na": False,
        }

    def test_server_errors_do_not_block_the_cell(self, magic, shell, monkeypatch, capsys):
        shell.user_ns["ran_conn"] = False
        _capture_lint(monkeypatch, magic, error=LinterConnectionError("down"))
        assert magic.lintcell("", "ran_conn = True\n2 + 2") == 4
        assert shell.user_ns["ran_conn"] is True
        assert "Cannot reach server:" in capsys.readouterr().out

        shell.user_ns["ran_http"] = False
        _capture_lint(
            monkeypatch,
            magic,
            error=LinterHTTPError(500, "Internal Server Error", "nope"),
        )
        assert magic.lintcell("", "ran_http = True\n3") == 3
        assert shell.user_ns["ran_http"] is True
        assert "HTTP 500 Internal Server Error" in capsys.readouterr().out

        _capture_lint(monkeypatch, magic, response={})
        assert magic.lintcell("", "4") == 4
        assert "Failed to read linter output" in capsys.readouterr().out

    def test_parse_errors_skip_the_cell(self, magic, shell, monkeypatch):
        shell.user_ns["cell_guard"] = 0
        captured = _capture_lint(monkeypatch, magic)

        with pytest.raises(ValueError, match="not valid"):
            magic.lintcell("--ip nope", "cell_guard = 1")
        with pytest.raises(ValueError, match="Invalid port"):
            magic.lintcell("--port 99999", "cell_guard = 1")
        with pytest.raises(argparse.ArgumentError):
            magic.lintcell("extra junk", "cell_guard = 1")
        with pytest.raises(argparse.ArgumentError):
            magic.lintcell("--port", "cell_guard = 1")

        assert shell.user_ns["cell_guard"] == 0
        assert "calls" not in captured

    def test_refresh_failure_skips_the_cell(self, magic, shell, monkeypatch):
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
        shell.register_magics(DataLinterMagic)
        registered = shell.find_cell_magic("lintcell").__self__
        registered.all_tracked_variables.clear()
        _capture_lint(monkeypatch, registered, response={"linting_output": "via shell"})

        assert shell.run_cell_magic("lintcell", "--show-stats", "10 - 3") == 7
