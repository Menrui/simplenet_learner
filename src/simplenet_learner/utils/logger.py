import logging


def get_logger(name=__name__) -> logging.Logger:
    logger = logging.getLogger(name)

    logging.getLogger("PIL.PngImagePlugin").setLevel(logging.CRITICAL + 1)
    logging.getLogger("PIL.TiffImagePlugin").setLevel(logging.CRITICAL + 1)
    logging.getLogger("matplotlib").setLevel(logging.CRITICAL + 1)

    return logger


def get_stream_logger(name=__name__) -> logging.Logger:
    logging.getLogger("PIL.PngImagePlugin").setLevel(logging.CRITICAL + 1)
    logging.getLogger("PIL.TiffImagePlugin").setLevel(logging.CRITICAL + 1)
    logging.getLogger("matplotlib").setLevel(logging.CRITICAL + 1)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    ch.setFormatter(formatter)

    log = logging.getLogger(name)
    log.setLevel(logging.DEBUG)
    log.handlers = []  # No duplicated handlers
    log.propagate = False  # workaround for duplicated logs in ipython
    log.addHandler(ch)
    return log
