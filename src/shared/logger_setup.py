from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


def setup_logger(
    module_name: str,
    log_dir: str,
    level: str = "INFO",
    log_to_console: bool = True,
    filename_prefix: Optional[str] = None,
) -> logging.Logger:
    """
    Crea y devuelve un logger con:
    - FileHandler
    - StreamHandler opcional
    - formato uniforme
    - carpeta de logs creada automáticamente

    Args:
        module_name: nombre lógico del módulo (ej: 'london_bot')
        log_dir: carpeta donde se guardará el log
        level: DEBUG, INFO, WARNING, ERROR, CRITICAL
        log_to_console: si True, también imprime en consola
        filename_prefix: prefijo opcional del nombre del archivo

    Returns:
        logging.Logger
    """
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d")
    prefix = filename_prefix or module_name
    file_name = f"{prefix}_{timestamp}.log"
    full_log_path = log_path / file_name

    logger = logging.getLogger(module_name)

    # Evita handlers duplicados si vuelves a llamar setup_logger
    if logger.handlers:
        return logger

    log_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(log_level)
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Handler a archivo
    file_handler = logging.FileHandler(full_log_path, encoding="utf-8")
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Handler a consola
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    logger.info(f"Logger inicializado | módulo={module_name} | archivo={full_log_path}")
    return logger