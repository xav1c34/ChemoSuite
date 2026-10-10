"""
modules — Вычислительные и хемометрические модули платформы ChemoSuite.
======================================================================
- fticr_core: Векторное ядро масс-спектрометрии NOM (FT-ICR MS).
- eem_core: Аналитическое ядро флуоресцентной спектроскопии (EEM-PARAFAC).
- chemo_ml: Многоблочный Data Fusion и хемометрика (PLS-DA, OPLS-DA).
- chemo_pubchem: Интеграция с PubChem, ChEMBL и HMDB REST API.
- project_io: Сериализация и восстановление проектов ChemoSuite (.chemo / ZIP).
- report_generator: Генератор аналитических паспортов Excel (.xlsx).
- ftir_core: Аналитическое ядро колебательной спектроскопии (ATR-FTIR & Raman).
- nmr_core: Аналитическое ядро спектроскопии ЯМР (13C / 1H NMR).
- thermal_core: Аналитическое ядро термического анализа и пиролиза (Py-GC-MS & TGA/DTG/DSC).
- sec_core: Аналитическое ядро эксклюзионной хроматографии (SEC / GPC).
"""

from . import fticr_core
from . import eem_core
from . import chemo_ml
from . import chemo_pubchem
from . import project_io
from . import report_generator
from . import ftir_core
from . import nmr_core
from . import thermal_core
from . import sec_core

__all__ = [
    "fticr_core",
    "eem_core",
    "chemo_ml",
    "chemo_pubchem",
    "project_io",
    "report_generator",
    "ftir_core",
    "nmr_core",
    "thermal_core",
    "sec_core",
]
