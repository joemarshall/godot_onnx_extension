from pathlib import Path
from enum import Enum
from dataclasses import dataclass


class TokenType(Enum):
    TK_CURLY_BRACKET_OPEN = "{"
    TK_CURLY_BRACKET_CLOSE = "}"
    TK_BRACKET_OPEN = "["
    TK_BRACKET_CLOSE = "]"
    TK_PARENTHESIS_OPEN = "("
    TK_PARENTHESIS_CLOSE = ")"
    TK_IDENTIFIER = "identifier"
    TK_STRING = "string"
    TK_STRING_NAME = "string_name"
    TK_NUMBER = "number"
    TK_COLOR = "color"
    TK_COLON = ":"
    TK_COMMA = ","
    TK_PERIOD = "."
    TK_EQUAL = "="
    TK_EOF = "EOF"
    TK_ERROR = "ERROR"


TokenTypeReverseMap = {x.value: x for x in TokenType if len(x.value) == 1}


@dataclass
class Token:
    tok_type: TokenType
    tok_value: str | int | float | None


def read(fname):
    p = Path(fname)
    extension_code = p.read_text()
    tokens = _tokenize(extension_code)
    sections = _split_sections(tokens)
    return sections


def _tokenize(text):
    token_list = []
    cur_token = None
    pos = 0
    while pos < len(text):
        char = text[pos]
        # single characters are mapped directly to tokens
        if char in TokenTypeReverseMap:
            token_list.append(Token(TokenTypeReverseMap[char], None))
            pos += 1
            continue
        match char:
            case ";":
                # ignore from ; to end of line
                while pos < len(text) and text[pos] != "\n":
                    pos += 1
            case '"':
                # read string token
                str_value = ""
                pos += 1
                while pos < len(text) and str_value != None:
                    char = text[pos]
                    match char:
                        case '"':
                            token_list.append(Token(TokenType.TK_STRING, str_value))
                            str_value = None
                        case "\\":
                            pos += 1
                            if pos >= len(text):
                                raise EOFError()
                            char = text[pos]
                            match char:
                                case "b":
                                    str_value += "\b"
                                case "n":
                                    str_value += "\n"
                                case "t":
                                    str_value += "\t"
                                case "\\":
                                    str_value += "\\"
                                case '"':
                                    str_value += '"'
                                case "'":
                                    str_value += "'"
                                case _:
                                    raise "Need to implement matching for hex sequences etc."
                        case _:
                            str_value += char
                    pos += 1
            case ch if ch.isnumeric() or ch in "-+":
                # parse number
                number_str = char
                pos += 1
                while pos < len(text) and (
                    text[pos].isnumeric() or text[pos] == "." or text[pos] == "e"
                ):
                    number_str += text[pos]
                    pos += 1
                try:
                    int_val = int(number_str)
                    token_list.append(Token(TokenType.TK_NUMBER, int_val))
                    continue
                except ValueError:
                    try:
                        float_val = float(number_str)
                        token_list.append(Token(TokenType.TK_NUMBER, float_val))
                        continue
                    except ValueError:
                        print("Couldnt parse number: {number_str}")
            case ch if ch.isspace():
                # ignore whitespace
                pos += 1
            case ch if ch.isidentifier():
                # read identifier text
                identifier_name = char
                pos += 1
                while pos < len(text) and (
                    text[pos].isidentifier()
                    or text[pos] == "."
                    or text[pos].isnumeric()
                ):
                    identifier_name += text[pos]
                    pos += 1
                token_list.append(Token(TokenType.TK_IDENTIFIER, identifier_name))
            case _:
                # bad character
                # ignore for now
                print(f"Warning character {char} in config file ignored")
                pos += 1
    return token_list


# parse a single value, and return the number of tokens used
def _parse_value(tokens):
    pos = 0
    match tokens[0].tok_type:
        case TokenType.TK_STRING:
            return (tokens[0].tok_value, 1)
        case TokenType.TK_NUMBER:
            return (tokens[0].tok_value, 1)
        case TokenType.TK_CURLY_BRACKET_OPEN:
            # read a map which is { Str : valstr }
            child_map = {}
            pos = 1
            while pos < len(tokens):
                if tokens[pos].tok_type == TokenType.TK_CURLY_BRACKET_CLOSE:
                    return (child_map, pos + 1)
                elif (
                    pos < len(tokens) - 2
                    and tokens[pos].tok_type == TokenType.TK_STRING
                    and tokens[pos + 1].tok_type == TokenType.TK_COLON
                ):
                    child_value, child_tokens = _parse_value(tokens[pos + 2 :])
                    child_map[tokens[pos].tok_value] = child_value
                    pos += child_tokens + 2
                else:
                    print("Couldn't parse map {} syntax", child_map, pos, tokens)


def _parse_section(tokens):
    pos = 0
    values = {}
    while pos < len(tokens):
        t = tokens[pos]
        # hopefully identifier = something pair
        if (
            pos < len(tokens) - 1
            and t.tok_type == TokenType.TK_IDENTIFIER
            and tokens[pos + 1].tok_type == TokenType.TK_EQUAL
        ):
            id_name = t.tok_value
            # parse value
            pos += 2
            value, num_tokens = _parse_value(tokens[pos:])
            pos += num_tokens
            values[id_name] = value
    return values


def _split_sections(tokens):
    sections = []
    cur_section = []
    cur_section_name = None
    pos = 0
    while pos < len(tokens):
        t = tokens[pos]
        if (
            t.tok_type == TokenType.TK_BRACKET_OPEN
            and tokens[pos + 1].tok_type == TokenType.TK_IDENTIFIER
            and tokens[pos + 2].tok_type == TokenType.TK_BRACKET_CLOSE
        ):
            if cur_section_name is not None:
                sections.append((cur_section_name, cur_section))
            cur_section_name = tokens[pos + 1].tok_value
            cur_section = []
            pos += 3
        else:
            cur_section.append(t)
            pos += 1
    if cur_section_name is not None:
        sections.append((cur_section_name, cur_section))
    return [(s, _parse_section(v)) for s, v in sections]


def merge_configurations(all_file_datas):
    # preserve section ordering from first file which each section is in
    section_order_list = []
    section_map = {}
    for file_data in all_file_datas:
        section_names = [x for x, y in file_data]
        last_section = None
        section_pos = 0
        while section_pos < len(section_names):
            last_section = None
            if section_pos > 0:
                last_section = section_names[section_pos - 1]
            next_section = None
            if section_pos + 1 < len(section_names):
                next_section = section_names[section_pos + 1]
            s = section_names[section_pos]
            if s in section_order_list:
                # already exists, ignore
                pass
            else:
                if last_section is not None and last_section in section_order_list:
                    # if previous section already exists, put it after it
                    section_order_list.insert(
                        section_order_list.index(last_section) + 1, s
                    )
                elif next_section is not None and next_section in section_order_list:
                    # if next section already exists, put it before it
                    section_order_list.insert(section_order_list.index(next_section), s)
                else:
                    # put at end of list
                    section_order_list.append(s)
            section_pos += 1
    # now merge all the data in a map
    section_map = {}
    for file_data in all_file_datas:
        for sname, svalues in file_data:
            if sname in section_map:
                section_map[sname] |= svalues
            else:
                section_map[sname] = svalues
    # now return sections in correct order
    return [(x, section_map[x]) for x in section_order_list]


def _escape_string(v):
    if isinstance(v,Path):
        v = str(v)
        v = v.replace('\\','/')
    v = v.replace("\\", "\\\\")
    v = v.replace("\n", "\\n")
    v = v.replace("\b", "\\b")
    v = v.replace("\t", "\\t")
    v = v.replace('"', '\\"')
    v = v.replace("'", "\\'")
    return v


def _format_value(v):
    if type(v) == int or type(v) == float:
        return str(v)
    elif type(v) == str or isinstance(v,Path):
        return f'"{_escape_string(v)}"'
    elif type(v) == dict:
        out_str = "{\n"
        for i, iv in v.items():
            out_str += f'    "{_escape_string(i)}" : {_format_value(iv)}\n'
        out_str += "}"
        return out_str


def get_as_text(section_list):
    all_text = ""
    for s, values in section_list:
        all_text += f"[{s}]\n\n"
        for k, v in values.items():
            all_text += f"{k} = {_format_value(v)}\n"
        all_text += "\n"
    return all_text


if __name__ == "__main__":
    fd = read("project\\addons\\onnx\\onnx.gdextension")
    print(merge_configurations([fd, fd]))
    print(get_as_text(fd))
