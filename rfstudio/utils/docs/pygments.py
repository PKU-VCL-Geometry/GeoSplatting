import re

from pygments.lexer import Lexer, RegexLexer, bygroups, do_insertions, include, line_re
from pygments.token import Comment, Generic, Keyword, Name, Number, Operator, Punctuation, String, Text, Whitespace
from pygments.util import shebang_matches

__all__ = ['PythonShellLexer', 'PythonSessionLexer']


class PythonShellLexer(RegexLexer):

    """Simple lexer for Python Shell."""

    name = 'Python Shell'
    aliases = ['pshell']
    filenames = ['*.pshell']
    mimetypes = ['text/pshell']

    tokens = {
        'root': [
            include('basic'),
            (r'`', String.Backtick, 'backticks'),
            include('data'),
            include('interp'),
        ],
        'interp': [
            (r'\$\(\(', Keyword, 'math'),
            (r'\$\(', Keyword, 'paren'),
            (r'\$\{#?', String.Interpol, 'curly'),
            (r'\$[a-zA-Z_]\w*', Name.Variable),  # user variable
            (r'\$(?:\d+|[#$?!_*@-])', Name.Variable),      # builtin
            (r'\$', Text),
        ],
        'basic': [
            (r'\b(if|fi|else|while|in|do|done|for|then|return|function|case|'
             r'select|break|continue|until|esac|elif)(\s*)\b',
             bygroups(Keyword, Whitespace)),
            (r'\b(alias|bg|bind|builtin|caller|cd|command|compgen|'
             r'complete|declare|dirs|disown|echo|enable|eval|exec|exit|'
             r'export|false|fc|fg|getopts|hash|help|history|jobs|kill|let|'
             r'local|logout|popd|printf|pushd|pwd|read|readonly|set|shift|'
             r'shopt|source|suspend|test|time|times|trap|true|type|typeset|'
             r'ulimit|umask|unalias|unset|wait)(?=[\s)`])',
             Name.Builtin),
            (r'^(pip|conda|git)(\s*)(\w+)(?=[\s)`])', bygroups(Name.Builtin, Whitespace, Name.Variable)),
            (r'^(python|mkdir|wget|unzip|rm)(?=[\s)`])', Name.Builtin),
            (r'\A#!.+\n', Comment.Hashbang),
            (r'#.*\n', Comment.Single),
            (r'\\[\w\W]', String.Escape),
            (r'(\s*)(\-[\w\-\_\d\.]+=)', bygroups(Whitespace, Operator)),
            (r'(\s*)(\-[\w\-\_\d\.]+)(?=[\s)`])', bygroups(Whitespace, Operator)),
            (r'\b(\w+)(=)([\d\.]+)(?=[\s)`])', bygroups(Name.Variable, Operator, Number)),
            (r'\b(\w+)(==)([\d\.]+)(?=[\s)`])', bygroups(Name.Variable, Operator, Number)),
            (r'[\[\]{}()=]', Operator),
            (r'<<<', Operator),  # here-string
            (r'<<-?\s*(\'?)\\?(\w+)[\w\W]+?\2', String),
            (r'&&|\|\|', Operator),
        ],
        'data': [
            (r'(?s)\$?"(\\.|[^"\\$])*"', String.Double),
            (r'"', String.Double, 'string'),
            (r"(?s)\$'(\\\\|\\[0-7]+|\\.|[^'\\])*'", String.Single),
            (r"(?s)'.*?'", String.Single),
            (r';', Punctuation),
            (r'&', Punctuation),
            (r'\|', Punctuation),
            (r'\s+', Whitespace),
            (r'\d+\b', Number),
            (r'[^=\s\[\]{}()$"\'`\\<&|;]+', Text),
            (r'<', Text),
        ],
        'string': [
            (r'"', String.Double, '#pop'),
            (r'(?s)(\\\\|\\[0-7]+|\\.|[^"\\$])+', String.Double),
            include('interp'),
        ],
        'curly': [
            (r'\}', String.Interpol, '#pop'),
            (r':-', Keyword),
            (r'\w+', Name.Variable),
            (r'[^}:"\'`$\\]+', Punctuation),
            (r':', Punctuation),
            include('root'),
        ],
        'paren': [
            (r'\)', Keyword, '#pop'),
            include('root'),
        ],
        'math': [
            (r'\)\)', Keyword, '#pop'),
            (r'\*\*|\|\||<<|>>|[-+*/%^|&<>]', Operator),
            (r'\d+#[\da-zA-Z]+', Number),
            (r'\d+#(?! )', Number),
            (r'0[xX][\da-fA-F]+', Number),
            (r'\d+', Number),
            (r'[a-zA-Z_]\w*', Name.Variable),  # user variable
            include('root'),
        ],
        'backticks': [
            (r'`', String.Backtick, '#pop'),
            include('root'),
        ],
    }

    def analyse_text(text):
        if shebang_matches(text, r'(ba|z|)sh'):
            return 1
        if text.startswith('$ '):
            return 0.2


class ShellSessionBaseLexer(Lexer):

    """
    Base lexer for shell sessions.

    .. versionadded:: 2.1
    """

    _bare_continuation = False
    _venv = re.compile(r'^(\([^)]*\))(\s*)')

    def get_tokens_unprocessed(self, text):
        innerlexer = self._innerLexerCls(**self.options)

        pos = 0
        curcode = ''
        insertions = []
        backslash_continuation = False

        for match in line_re.finditer(text):
            line = match.group()

            venv_match = self._venv.match(line)
            if venv_match:
                venv = venv_match.group(1)
                venv_whitespace = venv_match.group(2)
                insertions.append((len(curcode),
                                   [(0, Generic.Prompt.VirtualEnv, venv)]))
                if venv_whitespace:
                    insertions.append((len(curcode),
                                       [(0, Text, venv_whitespace)]))
                line = line[venv_match.end():]

            m = self._ps1rgx.match(line)
            if m:
                # To support output lexers (say diff output), the output
                # needs to be broken by prompts whenever the output lexer
                # changes.
                if not insertions:
                    pos = match.start()

                insertions.append((len(curcode),
                                   [(0, Generic.Prompt, m.group(1))]))
                curcode += m.group(2)
                backslash_continuation = curcode.endswith('\\\n')
            elif backslash_continuation:
                if line.startswith(self._ps2):
                    insertions.append((len(curcode),
                                       [(0, Generic.Prompt,
                                         line[:len(self._ps2)])]))
                    curcode += line[len(self._ps2):]
                else:
                    curcode += line
                backslash_continuation = curcode.endswith('\\\n')
            elif self._bare_continuation and line.startswith(self._ps2):
                insertions.append((len(curcode),
                                   [(0, Generic.Prompt,
                                     line[:len(self._ps2)])]))
                curcode += line[len(self._ps2):]
            else:
                if insertions:
                    toks = innerlexer.get_tokens_unprocessed(curcode)
                    for i, t, v in do_insertions(insertions, toks):
                        yield pos+i, t, v
                yield match.start(), Generic.Output, line
                insertions = []
                curcode = ''
        if insertions:
            for i, t, v in do_insertions(insertions,
                                         innerlexer.get_tokens_unprocessed(curcode)):
                yield pos+i, t, v


class PythonSessionLexer(ShellSessionBaseLexer):

    """Simple lexer for Python Shell."""

    name = 'Python Shell Session'
    aliases = ['psession']
    filenames = ['*.psession']
    mimetypes = ['text/psession']

    _innerLexerCls = PythonShellLexer
    _ps1rgx = re.compile(
        r'^((?:(?:\[.*?\])|(?:\(\S+\))?(?:| |sh\S*?|\w+\S+[@:]\S+(?:\s+\S+)' \
        r'?|\[\S+[@:][^\n]+\].+))\s*[$#%]\s*)(.*\n?)')
    _ps2 = '> '
