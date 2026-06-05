"""Verify that the recommender project Python environment is ready."""

import importlib


PACKAGES = {
    "numpy": "numpy",
    "pandas": "pandas",
    "sklearn": "scikit-learn",
    "matplotlib": "matplotlib",
    "seaborn": "seaborn",
    "jupyter": "jupyter",
}


def main() -> None:
    for module_name, package_name in PACKAGES.items():
        module = importlib.import_module(module_name)
        version = getattr(module, "__version__", "installed")
        print(f"{package_name}: {version}")

    print("Environment verification passed.")


if __name__ == "__main__":
    main()
