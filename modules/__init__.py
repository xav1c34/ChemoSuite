"""
modules — Вычислительные и хемометрические модули платформы ChemoSuite.
======================================================================
- fticr_core: Векторное ядро масс-спектрометрии NOM (FT-ICR MS).
- eem_core: Аналитическое ядро флуоресцентной спектроскопии (EEM-PARAFAC).
- chemo_ml: Многоблочный Data Fusion и хемометрика (PLS-DA, OPLS-DA).
- chemo_pubchem: Интеграция с PubChem, ChEMBL и HMDB REST API.
- project_io: Сериализация и восстановление проектов ChemoSuite (.chemo / ZIP).
- report_generator: Генератор аналитических паспортов Excel (.xlsx).
"""

from . import fticr_core
from . import eem_core
from . import chemo_ml
from . import chemo_pubchem
from . import project_io
from . import report_generator

__all__ = [
    "fticr_core",
    "eem_core",
    "chemo_ml",
    "chemo_pubchem",
    "project_io",
    "report_generator",
]
