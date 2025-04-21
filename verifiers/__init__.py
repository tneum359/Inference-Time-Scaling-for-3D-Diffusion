try:
    from .gemini_verifier import GeminiVerifier
except Exception as e:
    GeminiVerifier = None

try:
    from .laion_aesthetics import LAIONAestheticVerifier
except Exception as e:
    LAIONAestheticVerifier = None

SUPPORTED_VERIFIERS = {
    "gemini": GeminiVerifier if GeminiVerifier else None,
    "laion_aesthetic": LAIONAestheticVerifier if LAIONAestheticVerifier else None,
}

SUPPORTED_METRICS = {k: getattr(v, "SUPPORTED_METRIC_CHOICES", None) for k, v in SUPPORTED_VERIFIERS.items()}