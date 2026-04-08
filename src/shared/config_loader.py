from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"
ENVIRONMENTS_DIR = CONFIG_DIR / "environments"


def _read_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"YAML no encontrado: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"El YAML debe contener un diccionario en la raíz: {path}")
    return data


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_global_config() -> Dict[str, Any]:
    global_path = CONFIG_DIR / "global.yaml"
    return _read_yaml(global_path)


def load_environment_config(env_name: Optional[str] = None) -> Dict[str, Any]:
    global_cfg = load_global_config()
    active_env = env_name or global_cfg.get("active_environment")
    if not active_env:
        raise ValueError("No se encontró 'active_environment' en config/global.yaml")
    env_path = ENVIRONMENTS_DIR / f"{active_env}.yaml"
    env_cfg = _read_yaml(env_path)
    env_cfg["env_name"] = active_env
    return env_cfg


def load_module_config(module_name: str) -> Dict[str, Any]:
    module_path = CONFIG_DIR / f"{module_name}.yaml"
    return _read_yaml(module_path)


def load_app_config(module_name: str, env_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Combina:
    1) global.yaml               — defaults compartidos
    2) <module_name>.yaml        — defaults del módulo
    3) environments/<env>.yaml   — overrides específicos del entorno (capa final)

    Prioridad: env > module > global
    El entorno es la capa más específica y siempre gana.
    """
    global_cfg = load_global_config()
    env_cfg = load_environment_config(env_name=env_name)
    module_cfg = load_module_config(module_name)

    config = _deep_merge(global_cfg, module_cfg)   # module > global
    config = _deep_merge(config, env_cfg)           # env > module > global

    config["project_root"] = str(PROJECT_ROOT)
    config["config_dir"] = str(CONFIG_DIR)
    return config


if __name__ == "__main__":
    cfg = load_app_config("london_bot")
    print("Configuración cargada correctamente:\n")
    for key, value in cfg.items():
        print(f"{key}: {value}")