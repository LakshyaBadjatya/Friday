# © Lakshya Badjatya — Author
"""A redacting LLM wrapper: scrub PII from messages before a real provider sees them.

The privacy half of the security spine. When the owner points FRIDAY at a real
provider, :class:`RedactingLLM` wraps that provider and replaces high-confidence
PII (emails, payment-card numbers, IPv4 addresses, phone numbers) in each
outbound message's content with a typed placeholder, so personal data does not
leave the machine verbatim. Messages with no PII pass through unchanged.

It is a thin, drop-in :class:`~friday.providers.llm.LLMProvider` (like
:class:`~friday.providers.llm.FallbackLLM`): it depends only on the provider
contract plus the pure :class:`~friday.security.pii.PIIRedactor`, imports no LLM
SDK, and forwards ``tools`` / ``model`` untouched. Enabling it is a deliberate
privacy/utility trade-off — masked PII is also masked from the model — so it is
flag-gated and off by default.
"""

from __future__ import annotations

from friday.providers.llm import LLMProvider, LLMResponse, Message, ToolSpec
from friday.security.pii import PIIRedactor


class RedactingLLM(LLMProvider):
    """Wrap ``inner``, scrubbing PII from each message's content before sending.

    Args:
        inner: The provider that actually performs the completion.
        redactor: The :class:`PIIRedactor` to use (a default one if omitted).
    """

    def __init__(self, inner: LLMProvider, redactor: PIIRedactor | None = None) -> None:
        self._inner = inner
        self._redactor = redactor or PIIRedactor()

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        *,
        model: str | None = None,
    ) -> LLMResponse:
        """Scrub PII from each message, then delegate to the wrapped provider."""
        scrubbed = [self._scrub(message) for message in messages]
        return await self._inner.complete(scrubbed, tools, model=model)

    def _scrub(self, message: Message) -> Message:
        """Return ``message`` with PII masked in its content (unchanged if none)."""
        if message.content is None:
            return message
        cleaned = self._redactor.scrub(message.content)
        if cleaned == message.content:
            return message
        return message.model_copy(update={"content": cleaned})
