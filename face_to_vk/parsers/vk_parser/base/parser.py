from abc import abstractmethod


class BaseParser:
    """
    Base parser
    """

    MAX_VALUE = 1000  # error if more

    @property
    def METHOD(self):
        raise NotImplementedError

    @abstractmethod
    def _check_constraints(self, *args) -> bool:
        raise NotImplementedError

    @abstractmethod
    def parse_all(self) -> list:
        raise NotImplementedError
