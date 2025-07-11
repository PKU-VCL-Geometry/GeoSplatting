import importlib
import inspect
import re
from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable, List, Literal, Optional, Union, overload


@dataclass
class DocTreeNode():
    name: str
    doc: str
    source: str

    @abstractmethod
    def render(self, prefix: str) -> str:
        pass


@dataclass
class ParaNode:
    name: str
    doc: str
    typename: str
    default: Optional[str] = None
    required: bool = False
    positional: bool = False
    keyword_only: bool = False
    is_args: bool = False
    is_kwargs: bool = False


@dataclass
class RetNode:
    name: str
    doc: str
    typename: str


@dataclass
class ModuleNode(DocTreeNode):
    children: List[DocTreeNode]

    def render(self, prefix: str) -> str:
        return '\n'.join([child.render(prefix + '  ') for child in self.children])


@dataclass
class ContainerConstantNode(DocTreeNode):
    typename: Literal['list', 'dict', 'tuple', 'set']
    children: Iterable[Any]

    def render(self, prefix: str) -> str:
        raise NotImplementedError


@dataclass
class LiteralConstantNode(DocTreeNode):
    typename: Union[Literal['float', 'str', 'path', 'int', 'bool'], List[str]]
    value: Any

    def render(self, prefix: str) -> str:
        return (
            f'{prefix}  {self.name}: {self.typename} = {self.value!r}\n'
            f'{prefix}    {self.doc}\n'
        )


@dataclass
class FunctionNode(DocTreeNode):
    paras: List[ParaNode]
    ret: List[RetNode]

    def render(self, prefix: str) -> str:
        raise NotImplementedError


@dataclass
class MethodNode(FunctionNode):
    classname: str
    is_classmethod: bool = False
    is_abstractmethod: bool = False
    is_staticmethod: bool = False

    def render(self, prefix: str) -> str:
        raise NotImplementedError


@dataclass
class MemberNode(DocTreeNode):
    classname: str
    typename: str
    is_property: bool = False
    default: Optional[str] = None

    def render(self, prefix: str) -> str:
        raise NotImplementedError


@dataclass
class ClassNode(DocTreeNode):
    classname: str
    methods: List[MethodNode]
    members: List[MemberNode]
    basename: Optional[str] = None
    is_tensor: bool = False
    is_nn: bool = False
    is_dataclass: bool = False
    is_abc: bool = False

    def render(self, prefix: str) -> str:
        raise NotImplementedError


@overload
def make_doc_tree(target: ModuleType, *, root: bool = False, name: None = None) -> DocTreeNode:
    ...


@overload
def make_doc_tree(target: Any, *, root: bool = False, name: str) -> DocTreeNode:
    ...


def make_doc_tree(target: Any, *, root: bool = False, name: Optional[str] = None) -> DocTreeNode:
    if inspect.ismodule(target):
        children: List[DocTreeNode]
        children = []
        node = ModuleNode(
            name=target.__name__,
            doc=getattr(target, '__doc__', ''),
            source=inspect.getsource(target),
            children=children
        )
        if not root:
            return node
        if '__all__' in target.__dict__:
            exports = target.__all__
        else:
            exports = {
                k: v
                for k, v in target.__dict__.items()
                if not k.startswith('_') and not inspect.ismodule(v) and v.__module__ == target.__name__
            }
        if exports == {}:
            assert len(target.__path__) == 1, target.__path__
            path = Path(target.__path__[0])
            assert path.exists(), path
            for submodule_path in path.iterdir():
                if submodule_path.is_dir() and submodule_path.stem.startswith('_'):
                    continue
                if submodule_path.is_dir() or (submodule_path.is_file() and submodule_path.suffix == '.py'):
                    submodule = import_from_string(target.__name__ + '.' + submodule_path.stem)
                    exports[submodule_path.stem] = submodule
        for export_name, value in exports.items():
            children.append(make_doc_tree(value, name=export_name))
        return node
    assert name is not None
    return LiteralConstantNode(
        name=name,
        doc='',
        source='',
        typename='float',
        value=1.0,
    )


def import_from_string(import_str: str) -> Any:
    if '.' not in import_str:
        module_str, attr_str = import_str, None
    else:
        module_str, attr_str = import_str.rsplit('.', 1)

    try:
        module = importlib.import_module(module_str)
    except Exception:
        raise ValueError(f"Could not import {module_str!r}.")

    if attr_str is None:
        return module

    attr = getattr(module, attr_str, None)
    if attr is None:
        try:
            module = importlib.import_module(import_str)
        except ImportError:
            raise ValueError(f"Could not import {import_str!r}.")
        return module
    return attr


def get_params(signature: inspect.Signature) -> List[str]:
    """
    Given a function signature, return a list of parameter strings
    to use in documentation.

    Eg. test(a, b=None, **kwargs) -> ['a', 'b=None', '**kwargs']
    """
    params = []
    render_pos_only_separator = True
    render_kw_only_separator = True

    for parameter in signature.parameters.values():
        value = parameter.name
        if parameter.default is not parameter.empty:
            value = f"{value}={parameter.default!r}"

        if parameter.kind is parameter.VAR_POSITIONAL:
            render_kw_only_separator = False
            value = f"*{value}"
        elif parameter.kind is parameter.VAR_KEYWORD:
            value = f"**{value}"
        elif parameter.kind is parameter.POSITIONAL_ONLY:
            if render_pos_only_separator:
                render_pos_only_separator = False
                params.append("/")
        elif parameter.kind is parameter.KEYWORD_ONLY:
            if render_kw_only_separator:
                render_kw_only_separator = False
                params.append("*")
        params.append(value)

    return params


def trim_docstring(docstring: Optional[str]) -> str:
    """
    Trim leading indent from a docstring.

    See: https://www.python.org/dev/peps/pep-0257/#handling-docstring-indentation
    """
    if not docstring:
        return ""

    # Convert tabs to spaces (following the normal Python rules)
    # and split into a list of lines:
    lines = docstring.expandtabs().splitlines()
    # Determine minimum indentation (first line doesn't count):
    indent = 1000
    for line in lines[1:]:
        stripped = line.lstrip()
        if stripped:
            indent = min(indent, len(line) - len(stripped))

    # Remove indentation (first line is special):
    trimmed = [lines[0].strip()]
    if indent < 1000:
        for line in lines[1:]:
            trimmed.append(line[indent:].rstrip())

    # Strip off trailing and leading blank lines:
    while trimmed and not trimmed[-1]:
        trimmed.pop()
    while trimmed and not trimmed[0]:
        trimmed.pop(0)

    # Return a single string:
    return "\n".join(trimmed)


_RE = re.compile(r'(^|\n)([ ]*)\[\[\[(([a-zA-Z0-9_]+\.)*[a-zA-Z0-9_]+)\]\]\][ ]*($|\n)')


def on_page_markdown(md: str, **kwargs) -> str:
    result = md
    for m in _RE.finditer(md):
        result = md.replace(
            m.group(0),
            make_doc_tree(import_from_string(m.group(3)), root=True).render(prefix=m.group(2)),
        )
    return result
