"""Helper functionality for displaying pretty tables."""
from collections import OrderedDict
import json
import textwrap
import re

try:
    from itertools import izip_longest
except ImportError:
    from itertools import zip_longest as izip_longest


from .utils import to_unicode, is_number
join_lines = "\n".join


def get_schema(*dot_fields):
    """Convert a list of dot_fields into a nested schema."""
    schema = OrderedDict()
    for field in dot_fields:
        sub_schema = schema
        parts = [int(p) if p.isdigit() else p for p in re.split(r"[.\[\]]+", field) if p != ""]
        for piece in parts[:-1]:
            # It could be set explicitly to None, causing get(piece, {}) to return None
            sub_schema[piece] = sub_schema.get(piece) or OrderedDict()
            sub_schema = sub_schema[piece]

        # Now set this one to an empty dictionary
        sub_schema[parts[-1]] = None
    return schema


def headerspan(nested):
    """For a nested schema object, created the ordered headers, and float empty cells to the bottom."""
    headers = [[]]
    nested = nested

    for k in nested:
        if nested and nested.get(k):
            nested_headers = headerspan(nested[k])

            # Figure out how long it was by counting the leaf nodes
            span = len(nested_headers[-1])

            # add a merged column to the top of the nested headers
            nested_headers.insert(0, [(span, to_unicode(k))])

            # the prefix will be appended on top new rows that will be added
            prefix = [(s, '') for (s, _) in headers[0]]

            # walk from bottom to top, anchoring them to the bottom and moving up each time
            for pos, nested_header in enumerate(reversed(nested_headers), 1):
                # loop over each header in reverse order
                if pos > len(headers):
                    headers.insert(0, prefix[:])
                headers[-pos].extend(nested_header)
        else:
            span = 1
            # put the key on the bottom
            headers[-1].append((span, to_unicode(k)))

            # every row needs to grow at the top, depending on what's below it
            for next_i, header in enumerate(headers[:-1], 1):
                header.append((span, ''))

    return headers


def format_cell(cell, delim=", ", **kwargs):
    """Convert a cell to the rendered contents."""
    if isinstance(cell, (list, tuple)):
        return delim.join("{}".format(c) for c in cell)
    elif isinstance(cell, (dict, bool)):
        return json.dumps(cell)
    elif cell is None:
        return ""
    elif is_number(cell):
        return cell
    else:
        return to_unicode(cell)


def to_row(item, schema, **kwargs):
    """Convert an item to a row for a table."""
    row = []
    for k, subschema in schema.items():
        if isinstance(item, dict):
            cell = item.get(k)
        elif isinstance(item, list) and isinstance(k, int) and 0 <= k < len(item):
            cell = item[k]
        else:
            cell = None

        if not subschema:
            row.append(format_cell(cell, **kwargs))
        else:
            cells = [format_cell(c, **kwargs) for c in to_row(cell, schema[k])]
            row.extend(cells)
    return row


def _wrapped_lines(text, wrap):
    """Wrap lines while preserving original line breaks."""
    lines = []
    for line in text.rstrip().splitlines():
        lines.extend(textwrap.wrap(line, wrap))
    return lines


class Table(object):
    """Endgame pretty table base class."""

    def __init__(self, body, num_columns=0, names=None, merged_headers=None, top=True, bottom=True, border=True, pad=1,
                 border_div='=', col_sep='|', wrap=None, row_outline=None, outline=None, row_div='-', align=None):
        """Create a parameterized table with rows of cells."""
        # Create new rows based off the newlines
        array_body = []
        self._align = [align or '<'] * num_columns
        self._row_div = row_div
        self._border_div = border_div
        self._top = top
        self._bottom = bottom

        # Try to guess the alignment by looking at the first row
        if len(body) > 1:
            first = body[0]
            if num_columns == 0:
                num_columns = len(first)
                self._align = [align or '<'] * num_columns

            for i in range(num_columns):
                cell = first[i]
                if not align and is_number(cell):
                    self._align[i] = '>'
                    if isinstance(cell, float):
                        # convert all the rows to floats with nice decimals
                        for row in body:
                            if is_number(row[i]):
                                row[i] = "{:.3f}".format(row[i])

        if not wrap:
            for row in body:
                split_cells = [to_unicode(c).rstrip().splitlines() for c in row]
                array_body.append(list(izip_longest(*split_cells, fillvalue='')))
        else:
            for row in body:
                split_cells = [_wrapped_lines(to_unicode(c), wrap) for c in row]
                array_body.append(list(izip_longest(*split_cells, fillvalue='')))

        self._body = array_body

        # If the outline is set to None/auto, only add the outline if some of the rules span multiple lines
        # if row_outline is None and outline is None:
        #     if any(len(row) > 1 for row in array_body):
        #         row_outline = True

        self._outline = row_outline or outline
        self._padding = pad
        self._pad = ' ' * self._padding
        self._col_sep = col_sep if outline else ' '
        self._header_sep = self._col_sep.strip().center(len(self._col_sep))
        self._num_columns = num_columns
        self._border = border

        if merged_headers:
            self._headers = merged_headers
        elif names:
            self._headers = [[(1, k) for k in names]]
        else:
            self._headers = []

        self._widths = None
        self._row_width = None

    def calculate_widths(self):
        """Calculate the autofit widths for each column."""
        self._widths = [0] * self._num_columns
        for i in range(self._num_columns):
            try:
                self._widths[i] = max(len(line[i]) for row in self._body for line in row)
            except ValueError:
                pass

        delim_width = len(self._col_sep)

        # Now expand cells to accommodate for nested headers
        for header in self._headers:
            pos = 0
            for (span, k) in header:
                min_width = len(k)

                # update every row that this spans over
                num_delims = span - 1
                inner_widths = sum(self._widths[pos:pos + span])
                inner_padding = (self._padding * 2 + delim_width) * num_delims
                total_widths = inner_widths + inner_padding

                delta = min_width - total_widths

                if delta > 0:
                    add_each = int(delta / span)
                    add_one = delta % span
                    for i, column in enumerate(range(pos, pos + span)):
                        self._widths[column] += add_each
                        self._widths[column] += i < add_one
                pos += span

        self._row_width = (self._padding * 2 + delim_width) * self._num_columns + sum(self._widths) - delim_width

    def lines(self):
        """Get the lines in the table."""
        self.calculate_widths()
        delim_width = len(self._col_sep)

        join_row = self._col_sep.join
        join_header = self._header_sep.join if len(self._headers) > 1 else join_row
        rule_line = self._border_div * self._row_width
        row_div = self._row_div * self._row_width
        lines = []

        if self._border and self._top:
            lines.append(rule_line)

        for header in self._headers:
            pos = 0
            cells = []

            for (span, k) in header:  # type: (int, str)
                col_widths = sum(self._widths[pos:pos + span])
                num_delims = span - 1
                inner_padding = num_delims * self._padding * 2
                header_width = col_widths + inner_padding + (delim_width * num_delims)

                text = k.center(header_width) if span > 1 else k.ljust(header_width)
                cells.append(self._pad + text + self._pad)
                pos += span
            lines.append(join_header(cells))

        if len(self._headers):
            lines.append(rule_line)

        # header_format = col_sep.join('{:^' + str(width) + '}' for width in widths)

        # Now generate the format string for each row
        formats = []
        for i, (align, width) in enumerate(zip(self._align, self._widths)):
            if align == '<' and i == self._num_columns - 1:
                # No need to pad the right-most cell with empty space
                width = ""
            fmt = self._pad + '{:' + align + to_unicode(width) + "}" + self._pad
            formats.append(fmt)
        row_format = join_row(formats)
        format_row = row_format.format

        for i, row in enumerate(self._body):
            if i and self._outline:
                lines.append(row_div)
            lines.extend(format_row(*line) for line in row)

        if self._border and self._bottom:
            lines.append(rule_line)
        return lines

    @classmethod
    def from_list(cls, dot_fields, results, **params):  # type: (list[str], list[dict], str) -> Table
        """Generate a table from a list of results."""
        schema = get_schema(*dot_fields)
        num_columns = len(to_row({}, schema))
        body = [to_row(r, schema, **params) for r in results]
        headers = headerspan(schema)
        return Table(body, num_columns, merged_headers=headers, **params)

    def __iter__(self):
        """Iterate over the lines in the table."""
        return (line + "\n" for line in self.lines())

    def __unicode__(self):
        """Python 2 and 3 unicode support."""
        return join_lines(self.lines())

    def __str__(self):
        """Python 2 and 3 utf8 encoding."""
        unicoded = self.__unicode__()
        if not isinstance(unicoded, str):
            return unicoded.encode("utf-8")
        return unicoded
