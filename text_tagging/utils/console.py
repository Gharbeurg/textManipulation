from datetime import datetime
from colorama import Fore, Style, init

init(autoreset=True)


def _ts() -> str:
    return datetime.now().strftime("%d/%m/%Y %H:%M:%S")


def info(msg: str) -> None:
    print(f"{_ts()} - {msg}")


def success(msg: str) -> None:
    print(Fore.GREEN + f"[+] {msg}")


def error(msg: str) -> None:
    print(Fore.RED + f"[!] {msg}")


def stats(label: str, value: int) -> None:
    print(f"    {label} : {value}")
