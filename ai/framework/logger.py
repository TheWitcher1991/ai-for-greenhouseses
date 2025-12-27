from datetime import datetime


class Logger:
    @staticmethod
    def _log(level: str, message: str):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{ts}] [{level}] {message}")

    @staticmethod
    def info(message: str):
        Logger._log("INFO", message)

    @staticmethod
    def warning(message: str):
        Logger._log("WARN", message)

    @staticmethod
    def error(message: str):
        Logger._log("ERROR", message)


logger = Logger()
