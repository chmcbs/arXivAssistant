"""
Vector formatting helper for database storage
"""

def vector_literal(vector: list[float]) -> str:
    return "[" + ",".join(str(value) for value in vector) + "]"