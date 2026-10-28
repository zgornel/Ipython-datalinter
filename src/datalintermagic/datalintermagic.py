"""IPython magic commands for interacting with a DataLinter server.

This module provides line and cell magics that let Jupyter/IPython users
track pandas DataFrames or NumPy arrays and send them (as CSV) to a running
DataLinter HTTP service for linting.

Typical usage in a notebook::

    %load_ext datalintermagic
    %add --tracked-variable df --data-header
    %lintline df --show-stats
    %%lintcell --show-na
    # cell code that uses df
"""

import socket
import warnings
import ipaddress
import argparse
import shlex
import pandas as pd
import json
import http.client
import numpy as np

from IPython.core.magic import Magics, magics_class, line_magic, cell_magic
from IPython.display import display, Pretty
from termcolor import colored


DEFAULT_IP = "127.0.0.1"
DEFAULT_PORT = 10000
DEFAULT_TIMEOUT = 300
DEFAULT_DELIMITER = ','

# New exceptions
class LinterConnectionError(Exception):
    """Raised when the client cannot reach the DataLinter server or receives invalid JSON.

    This covers connection failures, timeouts, and responses that are not valid JSON.
    """

    def __init__(self, message):
        super().__init__(message)


class LinterHTTPError(Exception):
    """Raised when the DataLinter server returns a non-200 HTTP status.

    Attributes
    ----------
    status : int
        HTTP status code.
    reason : str
        HTTP reason phrase.
    body : str
        Response body (decoded text).
    """

    def __init__(self, status, reason, body):
        super().__init__(body)
        self.status = status
        self.reason = reason
        self.body = body


class DataLinterWarning(UserWarning):
    """Warning category used for all non-fatal messages emitted by the magics."""
    pass


def _parse_delimiter(value, default=','):
    """Return a validated delimiter string.

    Parameters
    ----------
    value : str or None
        User-supplied delimiter, or None to use the default.
    default : str, optional
        Fallback delimiter (default: ',').

    Returns
    -------
    str
        The delimiter to use.

    Raises
    ------
    ValueError
        If *value* is neither None nor a string.
    """
    if value is None:
        return default
    if isinstance(value, str):
        return value
    raise ValueError(f"expected str, got {value!r}")


@magics_class
class DataLinterMagic(Magics):
    """IPython magic class that tracks data variables and sends them to DataLinter.

    Line magics
    -----------
    %add
        Register a DataFrame or ndarray for later linting.
    %lintline
        Lint a single previously registered variable.

    Cell magic
    ----------
    %%lintcell
        Lint all currently tracked variables against the cell's source code,
        then execute the cell.
    """

    def __init__(self, shell):
        """Initialize the magic and its argument parsers.

        Parameters
        ----------
        shell : IPython.core.interactiveshell.InteractiveShell
            The IPython shell instance.
        """
        super().__init__(shell)
        self.all_tracked_variables = {}  # tracked variables, converted to CSV strings
        # Create %add parser
        self.parser_add = argparse.ArgumentParser(exit_on_error=False, add_help=False)
        self.parser_add.add_argument("--tracked-variable")
        self.parser_add.add_argument("--data-header", action='store_true')
        self.parser_add.add_argument("--data-delim")
        # Create %lintline and %lintcell parser
        self.parser_lint = argparse.ArgumentParser(exit_on_error=False, add_help=False)
        self.parser_lint.add_argument("tracked_variable", nargs='?', default=None)  # used only when linting lines
        self.parser_lint.add_argument("--ip")
        self.parser_lint.add_argument("--port")
        self.parser_lint.add_argument("--show-stats", action='store_true')
        self.parser_lint.add_argument("--show-na", action='store_true')
        self.parser_lint.add_argument("--show-passing", action='store_true')

    @staticmethod
    def display_warning(msg):
        """Emit a non-fatal warning via the warnings system.

        Parameters
        ----------
        msg : str
            Warning message.

        Returns
        -------
        None
        """
        warnings.warn(f"{msg}", category=DataLinterWarning)
        return None

    @staticmethod
    def display_output(results):
        """Pretty-print linting results in the notebook.

        Parameters
        ----------
        results : str
            The linting output text returned by the server.

        Returns
        -------
        None
        """
        header = "DataLinter:\n"
        display(Pretty(colored(f"{header}", "black", attrs=["bold"])))
        display(Pretty(colored(f"{results}", "black", attrs=["bold"])))
        return None

    def _parse_add_magic(self, line):
        """Parse the arguments of a %add magic line.

        Parameters
        ----------
        line : str
            The raw line after the magic name.

        Returns
        -------
        tuple or None
            (tracked_variable, data_header, data_delim) on success, or None
            if parsing failed (a warning is emitted).
        """
        try:
            args = self.parser_add.parse_args(shlex.split(line))
            # tracked variable checks
            if args.tracked_variable is None:
                DataLinterMagic.display_warning("Use '--tracked-variable' to specify a data variable!")
                return None
            elif str.isidentifier(args.tracked_variable) is False:
                DataLinterMagic.display_warning("Ivalid identifier specified with '--tracked-variable'")
                return None
            else:
                tracked_variable = args.tracked_variable
            # data header checks
            data_header = args.data_header
            # data delimiter checks
            data_delim = _parse_delimiter(args.data_delim)
            return tracked_variable, data_header, data_delim
        except Exception as ex:
            DataLinterMagic.display_warning(f"Error in %add magic line parser: {repr(ex)}")
            return None

    def _parse_lint_magic(self, line):
        """Parse the arguments of a %lintline or %%lintcell magic line.

        Parameters
        ----------
        line : str
            The raw line after the magic name.

        Returns
        -------
        tuple or None
            (tracked_variable, ip, port, show_stats, show_na, show_passing)
            on success, or None if parsing failed (a warning is emitted).
        """
        try:
            args = self.parser_lint.parse_args(shlex.split(line))
            tracked_variable = args.tracked_variable
            # ip
            try:
                _ip = ipaddress.ip_address(args.ip) if args.ip is not None else DEFAULT_IP
                ip = str(_ip)  # return string
            except Exception as exc:
                # Not legal
                raise ValueError(f"The IP '{args.ip!r}' is not valid") from exc
            # port
            try:
                port = int(args.port) if args.port is not None else DEFAULT_PORT
                assert 1 <= port <= 65535, "Invalid port value. Needs to be an integer between 1 and 65535"
            except Exception as ex:
                raise ValueError(f"{ex}")
            show_stats = args.show_stats
            show_na = args.show_na
            show_passing = args.show_passing
            return tracked_variable, ip, port, show_stats, show_na, show_passing
        except Exception as ex:
            DataLinterMagic.display_warning(f"Line parse error: {repr(ex)}")
            return None

    @line_magic
    def add(self, line):
        """Register a variable for linting.

        The variable must be a pandas DataFrame or a NumPy ndarray present in
        the user namespace. It is converted to a CSV string and stored together
        with its header and delimiter settings.

        Parameters
        ----------
        line : str
            Magic arguments, e.g. ``--tracked-variable df --data-header --data-delim ;``

        Returns
        -------
        None
        """
        parsed_args = self._parse_add_magic(line)
        if parsed_args is not None:
            tracked_variable, data_header, data_delim = parsed_args
            try:
                # Support only for: numpy.array and pandas.dataframe
                v = self.shell.user_ns[tracked_variable]
                v_delimited = self.to_csv_string(v, header=data_header, delimiter=data_delim, symbol=tracked_variable)
                self.all_tracked_variables[tracked_variable] = (v_delimited, data_header, data_delim)
            except Exception as ex:
                DataLinterMagic.display_warning(f"Could not add '{tracked_variable}' to linter data: {repr(ex)}")
        else:
            DataLinterMagic.display_warning("Could not parse line for '%add' magic")
        return None

    @line_magic
    def lintline(self, line):
        """Lint a single previously registered variable.

        The variable's current value is re-read from the user namespace (if it
        still exists) before being sent to the server.

        Parameters
        ----------
        line : str
            Magic arguments, e.g. ``df --ip 127.0.0.1 --port 10000 --show-stats``

        Returns
        -------
        None
        """
        parsed_args = self._parse_lint_magic(line)
        if parsed_args is not None:
            # Build request body for linter
            tracked_variable, ip, port, show_stats, show_na, show_passing = parsed_args
            if tracked_variable is not None and tracked_variable in self.all_tracked_variables.keys():
                # Update data variable from local namespace (uses the same delimiter, header)
                if tracked_variable in self.shell.user_ns:
                    _, _header, _delim = self.all_tracked_variables[tracked_variable]
                    v = self.shell.user_ns[tracked_variable]
                    self.all_tracked_variables[tracked_variable] = (self.to_csv_string(v, _header, _delim, tracked_variable), _header, _delim)
                _data, _header, _delim = self.all_tracked_variables[tracked_variable]
                varbody = {
                            'linter_input': {
                                'context': {
                                    'data': _data,
                                    'data_delim': _delim,
                                    'data_header': _header,
                                    'code': ''},
                                'options': {
                                    'show_stats': show_stats,
                                    'show_passing': show_passing,
                                    'show_na': show_na}
                            }
                        }
                jsonbody = json.dumps(varbody).encode("utf-8")
                try:
                    linter_response = self.http_lint_request(ip, port, jsonbody)
                    DataLinterMagic.display_output(linter_response['linting_output'])
                except LinterConnectionError as ex:
                    DataLinterMagic.display_warning(f"Cannot reach server: {repr(ex)}")
                except LinterHTTPError as ex:
                    DataLinterMagic.display_warning(
                        f"Server rejected the request "
                        f"(HTTP {ex.status} {ex.reason}):\n\t{ex.body}"
                    )
                except Exception as ex:
                    DataLinterMagic.display_warning(f"Failed to read linter output: {repr(ex)}")
            else:
                DataLinterMagic.display_warning("'%lintline' magic needs a valid data symbol")
        else:
            DataLinterMagic.display_warning("Could not parse line for '%lintline' magic")
        return None

    # Note: It would be interesting to Investigate when to execute the linter (before of after cell execution)
    #       One could provide two different magic commands i.e. @lint_before, @lint_after
    #       Currently, linting is done BEFORE cell execution and is non-blocking
    @cell_magic
    def lintcell(self, line, cell):
        """Lint all tracked variables against the cell source, then execute the cell.

        Linting is non-blocking: the cell always runs, even if parsing or the
        HTTP request fails. All tracked variables are refreshed from the user
        namespace before the request is sent.

        Parameters
        ----------
        line : str
            Magic arguments (IP, port, display flags). The optional positional
            variable name is ignored.
        cell : str
            The source code of the cell.

        Returns
        -------
        Any
            The result of executing the cell (``shell.run_cell(...).result``).
        """
        parsed_args = self._parse_lint_magic(line)
        # Build request body for linter
        if parsed_args is not None:
            _, ip, port, show_stats, show_na, show_passing = parsed_args
            # Check again for variable changes (re-bindings) and update all variables
            for tv in self.all_tracked_variables.keys():
                if tv in self.shell.user_ns:
                    _, _header, _delim = self.all_tracked_variables[tv]
                    v = self.shell.user_ns[tv]
                    self.all_tracked_variables[tv] = (self.to_csv_string(v, _header, _delim, tv), _header, _delim)
            varbody = {
                        'linter_input': {
                            'context': {
                                'data': self.all_tracked_variables,
                                # 'data_delim' and 'data_header' are not passed, each variable has its own
                                'code': cell},
                            'options': {
                                'show_stats': show_stats,
                                'show_passing': show_passing,
                                'show_na': show_na}
                        }
                    }
            jsonbody = json.dumps(varbody).encode("utf-8")
            try:
                linter_response = self.http_lint_request(ip, port, jsonbody)
                DataLinterMagic.display_output(linter_response['linting_output'])
            except LinterConnectionError as ex:
                DataLinterMagic.display_warning(f"Cannot reach server: {repr(ex)}")
            except LinterHTTPError as ex:
                DataLinterMagic.display_warning(
                    f"Server rejected the request "
                    f"(HTTP {ex.status} {ex.reason}):\n\t{ex.body}"
                )
            except Exception as ex:
                DataLinterMagic.display_warning(f"Failed to read linter output, perhaps linting failed: {repr(ex)}")
        else:
            DataLinterMagic.display_warning("Could not parse line for '%lintcell' magic")
        # Always run cell, regardless of linter status
        # Also, one can run with `self.shell.ex(cell)` if one only wants side effects
        result = self.shell.run_cell(cell)
        return result.result

    def to_csv_string(self, v, header, delimiter, symbol=None):
        """Convert a DataFrame or ndarray to a CSV string.

        Parameters
        ----------
        v : pandas.DataFrame or numpy.ndarray
            The value to convert.
        header : bool
            Whether to include a header row.
        delimiter : str
            Field separator.
        symbol : str, optional
            Variable name (used in warning messages).

        Returns
        -------
        str
            CSV representation.

        Raises
        ------
        TypeError
            If *v* is neither a DataFrame nor an ndarray.
        """
        if isinstance(v, pd.DataFrame):
            return self.dataframe_to_csv_string(v, header, delimiter, symbol=symbol)
        elif isinstance(v, np.ndarray):
            return self.ndarray_to_csv_string(v, header, delimiter, symbol=symbol)
        else:
            raise TypeError(f"Unsupported type '{type(v)}' for delimited-string transformation")

    def ndarray_to_csv_string(self, arr, header, delimiter, symbol=None):
        """Convert a NumPy ndarray to a CSV string via a temporary DataFrame.

        Parameters
        ----------
        arr : numpy.ndarray
            Input array.
        header : bool
            Whether to include a header row.
        delimiter : str
            Field separator.
        symbol : str, optional
            Variable name (used in warning messages).

        Returns
        -------
        str
            CSV representation.
        """
        df = pd.DataFrame(arr)
        return self.dataframe_to_csv_string(df, header=header, delimiter=delimiter, symbol=symbol)

    def dataframe_to_csv_string(self, df, header, delimiter, symbol=None):
        """Convert a pandas DataFrame to a CSV string suitable for DataLinter.

        Non-string column names cause a warning and are replaced by generated
        names (x0, x1, ...). Missing values are written as the string ``NA``.

        Parameters
        ----------
        df : pandas.DataFrame
            Input frame.
        header : bool
            Whether to include a header row.
        delimiter : str
            Field separator.
        symbol : str, optional
            Variable name (used in warning messages).

        Returns
        -------
        str
            CSV representation.
        """
        _header = None
        _colnames = list(df.columns)
        _str_colnames = _colnames and all(isinstance(col, str) for col in _colnames)
        if header is True and _str_colnames:
            _header = _colnames
        elif header is True and not _str_colnames:
            DataLinterMagic.display_warning(f"Column names of '{symbol}' will be auto-generated as x0,x1...")
            _header = ['x'+str(i) for i in range(len(df.columns))]
        else:
            # header is False, header not exported to csv
            pass
        return df.to_csv(None,
                         sep=delimiter,
                         header=_header,
                         index=False,
                         na_rep="NA")  # Linter supports ["", "NA", "NaN", "N/A", "NAN"]

    def http_lint_request(self, ip, port, body, timeout=DEFAULT_TIMEOUT, path="/api/lint"):
        """Send a JSON POST request to the DataLinter server and return the parsed response.

        Wildcard bind addresses (``0.0.0.0``, ``::``, ``[::]``) are rewritten to
        the loopback address for the actual connection. The HTTP connection
        timeout is always ``DEFAULT_TIMEOUT``; the *timeout* argument is only
        used in the timeout error message.

        Parameters
        ----------
        ip : str
            Server IP address.
        port : int or str
            Server port.
        body : bytes
            JSON request body.
        timeout : float, optional
            Timeout value used only in the error message (default: DEFAULT_TIMEOUT).
        path : str, optional
            HTTP path (default: ``/api/lint``).

        Returns
        -------
        dict
            Parsed JSON response.

        Raises
        ------
        LinterConnectionError
            On connection failure, timeout, or invalid JSON.
        LinterHTTPError
            On any non-200 HTTP status.
        """
        connect_host = DEFAULT_IP if ip in {"0.0.0.0", "::", "[::]"} else ip
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Content-Length": str(len(body)),
        }
        conn = http.client.HTTPConnection(connect_host, int(port), timeout=DEFAULT_TIMEOUT)
        try:
            conn.request("POST", path, body=body, headers=headers)
            response = conn.getresponse()
            raw = response.read()
            status, reason = response.status, response.reason
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
        try:
            text = raw.decode("utf-8", errors="replace")
            if status != 200:
                raise LinterHTTPError(status, reason, text)
            parsed = json.loads(text)
            return parsed
        except json.JSONDecodeError as ex:
            raise LinterConnectionError(f"Invalid JSON from server: {ex}") from ex
