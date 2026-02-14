import hashlib
from typing import Union


class HashFunction:
    """Hash function for feature engineering."""

    def __init__(self, hash_size: int = 1000000):
        """
        Args:
            hash_size: Size of the hash space
        """
        self.hash_size = hash_size

    def hash_str(self, s: str) -> int:
        """Hash a string to integer in [0, hash_size)."""
        return int(hashlib.md5(s.encode()).hexdigest(), 16) % self.hash_size

    def hash_string(self, s: str) -> int:
        """Hash a string to integer in [0, hash_size). Alias for hash_str."""
        return self.hash_str(s)

    def hash_feature(self, feature_value: Union[str, int], feature_index: int) -> int:
        """Hash a feature value with its index."""
        feature_str = f"{feature_index}_{feature_value}"
        return self.hash_str(feature_str)

    def __call__(self, feature_value: Union[str, int], feature_index: int) -> int:
        """Make the object callable."""
        return self.hash_feature(feature_value, feature_index)