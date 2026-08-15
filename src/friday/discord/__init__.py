"""Discord: the private room.

Two ways in. :mod:`friday.api.routes_discord` answers signed slash commands over
HTTP; :mod:`friday.discord.gateway` holds a WebSocket open so she can read normal
messages, chime in, and set the status line. :mod:`friday.discord.banter` is how
she talks once she has decided to.
"""
