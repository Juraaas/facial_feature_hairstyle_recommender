class PipelineError(Exception):
    def __init__(self, code: str, message: str):
        self.code = code
        self.message = message
        super().__init__(message)

INVALID_IMAGE = "INVALID_IMAGE"
NO_FACE_DETECTED = "NO_FACE_DETECTED"
FACE_TOO_SMALL = "FACE_TOO_SMALL"
FACE_ROTATED = "FACE_ROTATED"
FACE_TILTED = "FACE_TILTED"
POOR_ALIGNMENT = "POOR_ALIGNMENT"
POOR_LIGHTNING = "POOR_LIGHTNING"
INTERNAL_ERROR = "INTERNAL_ERROR"