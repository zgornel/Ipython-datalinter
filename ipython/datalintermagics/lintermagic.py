import socket
import argparse
import shlex
import IPython
import pandas as pd
import json
import http.client
import numpy as np

from IPython.core.magic import (Magics, magics_class, line_magic, cell_magic)

TRACKED_VARIABLES = {}  # tracked variables, converted to CSV strings
DEFAULT_IP = "127.0.0.1"
DEFAULT_PORT = 10000

# New exceptions
class LinterConnectionError(Exception):
    def __init__(self, message):
        super().__init__(message)

class LinterHTTPError(Exception):
    def __init__(self, status, reason, body):
        super().__init__(body)
        self.status = status
        self.reason = reason
        self.body = body

def _parse_delimiter(value, default=','):
    if value is None:
        return default
    if isinstance(value, str):
        return value
    raise ValueError(f"expected str, got {value!r}")

@magics_class
class DataLinterMagic(Magics):
    tracked_variable = None
    data_header = None
    data_delim = None
    ip = DEFAULT_IP
    port = DEFAULT_PORT
    show_stats = False
    show_na = False
    show_passing = False

    def __parse_add_magic(self, line):
        try:
            parser = argparse.ArgumentParser()
            parser.add_argument("--tracked-variable")
            parser.add_argument("--data-header", action='store_true')
            parser.add_argument("--data-delim")
            args = parser.parse_args(shlex.split(line))
            # tracked variable checks
            if args.tracked_variable is None:
                print("Warning (Linter): Use '--tracked-variable' to specify a data variable!")  # noqa: E501
                return None
            tracked_variable = args.tracked_variable
            # data header checks
            data_header = args.data_header
            # data delimiter checks
            data_delim = _parse_delimiter(args.data_delim)
            return tracked_variable, data_header, data_delim
        except Exception as ex:
            print(f"Warning (Linter): Could not parse line for %%add line magic:\n\t{ex}")  # noqa: E501
            return None

    def __parse_lint_magic(self, line):
        try:
            parser = argparse.ArgumentParser()
            parser.add_argument("tracked_variable", nargs='?', default=None)  # used only when linting lines
            parser.add_argument("--ip")
            parser.add_argument("--port")
            parser.add_argument("--show-stats", action='store_true')
            parser.add_argument("--show-na", action='store_true')
            parser.add_argument("--show-passing", action='store_true')
            args = parser.parse_args(shlex.split(line))
            tracked_variable = args.tracked_variable
            # IP
            try:
                ip = args.ip if args.ip is not None else DEFAULT_IP
                socket.inet_aton(ip) # check legal ip
            except socket.error:
                # Not legal
                raise ValueError(f"The IP '{ip!r}' is not valid")
            try:
                port = int(args.port) if args.port is not None else DEFAULT_PORT
                assert port > 0 and port < 65535, "Invalid port value. Needs to be an integer between 1 and 65535"
            except Exception as ex:
                raise ValueError(f"{ex}")
            show_stats = args.show_stats
            show_na = args.show_na
            show_passing = args.show_passing
            return tracked_variable, ip, port, show_stats, show_na, show_passing
        except Exception as ex:
            print(f"Warning (Linter): Could not parse line for linting cell magic:\n\t{ex}")  # noqa: E501
            return None

    @line_magic
    def add(self, line):
        parsed_args = self.__parse_add_magic(line)
        ipy = IPython.get_ipython()
        if parsed_args is not None:
            tracked_variable, data_header, data_delim = parsed_args  # noqa: E501
            try:
                # Support only for: numpy.array and pandas.dataframe
                v = ipy.ev(tracked_variable)
                v_delimited = self.to_csv_string(v, header=data_header, delimiter=data_delim)
                TRACKED_VARIABLES[tracked_variable] = (v_delimited, data_header, data_delim)
                print(f">>DEBUG: Added '{tracked_variable}' as delimited string to tracked variables.")  # noqa: E500
            except Exception as ex:
                print(f"Warning (Linter): Could not add '{tracked_variable}' to linter data:\n\t{ex}")  # noqa: E501
        return None

    @line_magic
    def lintline(self, line):
        parsed_args = self.__parse_lint_magic(line)
        ipy = IPython.get_ipython()
        if parsed_args is not None:
            for tv in TRACKED_VARIABLES.keys():
                if tv not in self.shell.user_ns:
                    v, header, delimiter = self.shell.user_ns[tv]
                    TRACKED_VARIABLES[tv] = self.to_csv_string(v, header, delimiter)
            # Build request body for linter
            tracked_variable, self.ip, self.port, self.show_stats, self.show_na, \
                self.show_passing = parsed_args
            if tracked_variable in TRACKED_VARIABLES.keys():
                _data, _header, _delim = TRACKED_VARIABLES[tracked_variable]
                varbody = {
                            'linter_input': {
                                'context': {
                                    'data':_data,
                                    'data_delim': _delim,
                                    'data_header': _header,
                                    'code': ''},
                                'options': {
                                    'show_stats': self.show_stats,
                                    'show_passing': self.show_passing,
                                    'show_na': self.show_na}
                            }
                        }
                jsonbody = json.dumps(varbody).encode("utf-8")
                try:
                    linter_response = self.http_lint_request(self.ip, self.port, jsonbody)
                    print(f"Linter output\n-------------\n{linter_response['linting_output']}")
                except LinterConnectionError as ex:
                    print(f"Warning (Linter): Cannot reach server:\n\t{ex}")
                except LinterHTTPError as ex:
                    print(
                        f"Warning (Linter): Server rejected the request "
                        f"(HTTP {ex.status} {ex.reason}):\n\t{ex.body}"
                    )
                except Exception as ex:
                    print(f"Warning (Linter): Failed to read linter output (perhaps linting failed):\n\t{ex}")  # noqa: E501
            else:
                print(f">>DEBUG: '%lintline' magic FAILED (parsed_args={parsed_args})")  # noqa: E501
        return None

    # Note: It would be interesting to Investigate when to execute the linter (before of after cell execution)
    #       One could provide two different magic commands i.e. @lint_before, @lint_after
    #       Currently, linting is done BEFORE cell execution and is non-blocking
    @cell_magic
    def lintcell(self, line, cell):
        parsed_args = self.__parse_lint_magic(line)
        ipy = IPython.get_ipython()
        # Check again for variable changes (re-bindings)
        for tv in TRACKED_VARIABLES.keys():
            if tv not in self.shell.user_ns:
                v, header, delimiter = self.shell.user_ns[tv]
                TRACKED_VARIABLES[tv] = self.to_csv_string(v, header, delimiter)
        # Build request body for linter
        if parsed_args is not None:
            _, self.ip, self.port, self.show_stats, self.show_na, \
                self.show_passing = parsed_args
            varbody = {
                        'linter_input': {
                            'context': {
                                'data': TRACKED_VARIABLES,
                                'data_delim': self.data_delim,
                                'data_header': self.data_header,
                                'code': cell},
                            'options': {
                                'show_stats': self.show_stats,
                                'show_passing': self.show_passing,
                                'show_na': self.show_na}
                        }
                    }
            jsonbody = json.dumps(varbody).encode("utf-8")
            try:
                linter_response = self.http_lint_request(self.ip, self.port, jsonbody)
                print(f"Linter output\n-------------\n{linter_response['linting_output']}")
            except LinterConnectionError as ex:
                print(f"Warning (Linter): Cannot reach server:\n\t{ex}")
            except LinterHTTPError as ex:
                print(
                    f"Warning (Linter): Server rejected the request "
                    f"(HTTP {ex.status} {ex.reason}):\n\t{ex.body}"
                )
            except Exception as ex:
                print(f"Warning (Linter): Failed to read linter output (perhaps linting failed):\n\t{ex}")  # noqa: E501
        else:
            print(f">>DEBUG: '%%lintcell' magic FAILED (parsed_args={parsed_args})")  # noqa: E501
        # Run cell (or self.shell.ex(cell) if you only want side effects)
        result = self.shell.run_cell(cell)
        return result.result

    def to_csv_string(self, v, header, delimiter):
        if isinstance(v, pd.DataFrame):
            return self.dataframe_to_csv_string(v, header, delimiter)
        elif isinstance(v, np.ndarray):
            return self.ndarray_to_csv_string(v, header, delimiter)
        else:
            raise TypeError(f"Unsupported type '{type(v)}' for delimited-string transformation")

    def ndarray_to_csv_string(self, arr, header, delimiter):
        df = pd.DataFrame(arr)
        return self.dataframe_to_csv_string(df, header=header, delimiter=delimiter)

    def dataframe_to_csv_string(self, df, header, delimiter):
        _header = None
        _colnames = list(df.columns)
        _str_colnames = isinstance(_colnames[0], str)
        if header is True and _str_colnames:
            _header = _colnames
        elif header is True and not _str_colnames:
            _header = ['x'+str(i) for i in range(len(df.columns))]
        else:
            # header if false, header exported to csv
            pass
        return df.to_csv(None,
                         sep=delimiter,
                         header=_header,
                         index=False)

    def http_lint_request(self, ip, port, body, timeout=300.0, path="/api/lint"):
        connect_host = DEFAULT_IP if ip in {"0.0.0.0", "::", "[::]"} else ip
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Content-Length": str(len(body)),
        }
        conn = http.client.HTTPConnection(connect_host, int(port), timeout=timeout)
        try:
            conn.request("POST", path, body=body, headers=headers)
            response = conn.getresponse()
            raw = response.read()
            status, reason = response.status, response.reason
            content_type = response.getheader("Content-Type") or ""
        except socket.timeout as ex:
            raise LinterConnectionError(
                f"Timed out after {timeout}s talking to {connect_host}:{port}"
            ) from ex
        except (ConnectionRefusedError, socket.gaierror, OSError) as ex:
            raise LinterConnectionError(
                f"Could not reach DataLinter at {connect_host}:{port} ({ex}). "
                "Is the Docker server running?"
            ) from ex
        finally:
            conn.close()
        text = raw.decode("utf-8", errors="replace")
        if status != 200:
            raise LinterHTTPError(status, reason, text)
        parsed = json.loads(text)
        return parsed
