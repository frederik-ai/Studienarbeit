from pygments.lexer import RegexLexer, bygroups
from pygments.token import *

class TFLexer(RegexLexer):
    name = 'TensorFlow'
    aliases = ['tf']
    filenames = ['*.tf']

    tokens = {
        'root': [
            # TensorFlow keywords
            (r'\b(tf|Tensor|Variable|Session)\b', Keyword),

            # TensorFlow functions
            (r'\b(tf\.[a-zA-Z_][a-zA-Z0-9_\.]*)\b', Name.Function),

            # Other Python syntax
            (r'(?<=\w)\.(?=\w)', Operator),
            (r'[^\S\n]+', Text),
            (r'#.*$', Comment.Single),
            (r'\"\"\"[\s\S]*?\"\"\"', String.Doc),
            (r'\"[^\n\"]*\"', String.Double),
            (r"\'[^\n\']*\'", String.Single),
            (r'\b(None|True|False)\b', Keyword.Constant),
            (r'\b(and|as|assert|break|class|continue|def|del|elif|else|except|finally|for|from|global|if|import|in|is|lambda|nonlocal|not|or|pass|raise|return|try|while|with|yield)\b', Keyword),
            (r'\b(abs|all|any|ascii|bin|bool|bytearray|bytes|callable|chr|classmethod|compile|complex|delattr|dict|dir|divmod|enumerate|eval|exec|filter|float|format|frozenset|getattr|globals|hasattr|hash|help|hex|id|input|int|isinstance|issubclass|iter|len|list|locals|map|max|memoryview|min|next|object|oct|open|ord|pow|print|property|range|repr|reversed|round|set|setattr|slice|sorted|staticmethod|str|sum|super|tuple|type|vars|zip)\b', Name.Builtin),
            (r'\b(NotImplemented|Ellipsis)\b', Name.Builtin.Pseudo),
            (r'[a-zA-Z_][a-zA-Z0-9_]*', Name),
            (r'[0-9]+', Number.Integer),
            (r'0[bB][01]+', Number.Bin),
            (r'0[oO][0-7]+', Number.Oct),
            (r'0[xX][a-fA-F0-9]+', Number.Hex),
            (r'[0-9]+\.[0-9]*(e[+-]?[0-9]+)?', Number.Float),
            (r'\.[0-9]+(e[+-]?[0-9]+)?', Number.Float),
            (r'[+\-*/%&|\^~<>=!]=?', Operator),
            (r'[\[\](){},;:@]', Punctuation),
        ]
    }