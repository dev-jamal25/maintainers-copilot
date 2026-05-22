class VaultError(Exception):
    """Base class for Vault adapter errors."""


class VaultConfigError(VaultError):
    """Vault bootstrap config (address or token) is missing or invalid."""


class VaultUnavailableError(VaultError):
    """Vault is unreachable, sealed, or not initialized."""


class VaultAuthenticationError(VaultError):
    """Vault rejected the supplied token."""


class VaultSecretNotFoundError(VaultError):
    """A KV v2 path returned no secret or no data envelope."""


class ModelServerError(Exception):
    """Base class for model-server adapter errors."""


class ModelServerUnavailableError(ModelServerError):
    """The model-server was unreachable, timed out, or kept returning 5xx after retries."""


class ModelServerResponseError(ModelServerError):
    """The model-server returned a 4xx or a response body the adapter could not parse."""


class LLMError(Exception):
    """Base class for LLM adapter errors."""


class LLMUnavailableError(LLMError):
    """The LLM provider was unreachable, timed out, rate-limited, or returned an error."""


class TracingError(Exception):
    """Base class for tracing adapter errors."""


class TracingConfigError(TracingError):
    """Tracing config (host, Vault secret shape, or required field) is missing or invalid."""


class TracingUnavailableError(TracingError):
    """The Langfuse client could not be initialized or did not pass its readiness check."""
