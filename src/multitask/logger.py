import logging

from tqdm import tqdm

log_format = "%(asctime)s - %(levelname)s - %(filename)s %(lineno)d - %(message)s"


class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.INFO):
        super().__init__(level)
        self.setFormatter(logging.Formatter(log_format))

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)


logging.basicConfig(
    level=logging.INFO, format=log_format, filename="multitask.log", filemode="w"
)
logger = logging.getLogger(__name__)
logger.addHandler(TqdmLoggingHandler())

logger.info("Logger initialized for multitask module.")
