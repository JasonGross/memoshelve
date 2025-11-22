import inspect
import ast
import types
import functools
from typing import Callable, Set, List, TypeVar

T = TypeVar("T")


def get_function_source(
    func: Callable,
    *,
    recursive: bool = False,
    map_fn: Callable[[str], T] = str,
    reduce_fn: Callable[[T, T], T] = lambda a, b: f"{a}\n{b}",
) -> T:
    """
    Retrieves the source code of a function and optionally its dependencies.

    This function uses AST parsing to find calls and resolves them against
    globals and closures to ensure the correct source is retrieved.

    Args:
        func: The function to analyze.
        recursive: If True, includes source for user-defined functions called by `func`.
        map_fn: Transformation to apply to each source string (e.g., hashing).
        reduce_fn: Aggregation function to combine results (e.g., concatenation).
                   Defaults to joining with a newline.

    Returns:
        The aggregated result of the source codes.

    Raises:
        ValueError: If the source code for the primary function cannot be retrieved.
    """

    # Track visited IDs to handle circular references (Recursion trap)
    visited_ids: Set[int] = set()
    collected_sources: List[str] = []

    def _collect_sources(target_func: Callable):
        # 1. Unwrap decorators (e.g., @functools.wraps) using standard lib
        try:
            real_func = inspect.unwrap(target_func)
        except Exception:
            real_func = target_func

        # Handle bound methods (e.g., MyClass().method)
        if inspect.ismethod(real_func):
            real_func = real_func.__func__

        # 2. Cycle and Validity Checks
        func_id = id(real_func)
        if func_id in visited_ids:
            return
        visited_ids.add(func_id)

        # 3. Attempt to get source
        try:
            source = inspect.getsource(real_func)
        except (OSError, TypeError):
            # Built-ins, C-extensions, or dynamic code often lack source files.
            # We skip these silently for dependencies.
            if len(visited_ids) == 1:
                # If this is the *primary* function requested, we must raise.
                raise ValueError(f"Source code not available for {target_func}")
            return

        collected_sources.append(source)

        # 4. If recursive, find dependencies via AST
        if recursive:
            dependencies = _find_called_functions(real_func, source)
            for dep in dependencies:
                _collect_sources(dep)

    # Start collection
    _collect_sources(func)

    # Apply Map
    mapped_values = [map_fn(s) for s in collected_sources]

    # Apply Reduce
    if not mapped_values:
        return map_fn("")

    if len(mapped_values) == 1:
        return mapped_values[0]

    return functools.reduce(reduce_fn, mapped_values)


def _find_called_functions(func: Callable, source: str) -> Set[Callable]:
    """
    Parses source code to find function calls and resolves them to objects.
    """
    dependencies = set()

    try:
        tree = ast.parse(inspect.cleandoc(source))
    except SyntaxError:
        return dependencies

    # 1. Extract names of all called functions
    called_names = set()

    class CallVisitor(ast.NodeVisitor):
        def visit_Call(self, node):
            # Standard call: func()
            if isinstance(node.func, ast.Name):
                called_names.add(node.func.id)
            # Method call: obj.method() - technically a dependency,
            # but usually hard to resolve statically without type inference.
            self.generic_visit(node)

    CallVisitor().visit(tree)

    # 2. Resolve names to actual objects (check Closures -> Globals)

    # Check Closures (nonlocals)
    closure_vars = inspect.getclosurevars(func)

    for name in called_names:
        obj = None

        # Order of resolution: Nonlocals -> Globals -> Builtins (skip builtins)
        if name in closure_vars.nonlocals:
            obj = closure_vars.nonlocals[name]
        elif name in closure_vars.globals:
            obj = closure_vars.globals[name]
        elif name in getattr(func, "__globals__", {}):
            obj = func.__globals__[name]

        if obj and isinstance(obj, (types.FunctionType, types.MethodType)):
            dependencies.add(obj)

    return dependencies
