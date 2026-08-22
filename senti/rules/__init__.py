"""
senti.rules
===========
Violation rules, self-registering.

Importing this package imports every rule module inside it, so
`Rule.__init_subclass__` fires and RULE_REGISTRY is populated. That is what
makes "drop a new file in this folder, enable it in YAML" actually work -- no
manual import list to maintain, and no dependency on the engine having been
imported first.
"""

import importlib
import pkgutil

from .base import (  # noqa: F401
    RULE_REGISTRY,
    Rule,
    Violation,
    available_rules,
    get_rule,
)


def _autoload() -> None:
    for mod in pkgutil.iter_modules(__path__):
        if mod.name != 'base':
            importlib.import_module(f'{__name__}.{mod.name}')


_autoload()

__all__ = ['Rule', 'Violation', 'RULE_REGISTRY', 'get_rule', 'available_rules']
