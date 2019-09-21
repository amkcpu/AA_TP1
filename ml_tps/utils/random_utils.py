import random
import string


def random_string(size: int) -> str:
    return ''.join(random.choice(string.ascii_letters) for _ in range(size))

