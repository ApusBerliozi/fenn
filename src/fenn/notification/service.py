from abc import ABC, abstractmethod


class Service(ABC):
    """Abstract base class for notification services."""

    @abstractmethod
    def send_notification(self, message: str) -> None:
        """Send a notification message.

        Args:
            message: The message to send.

        Raises:
            Exception: If the notification fails to send.
        """
        pass
