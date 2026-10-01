"""Exchange abstraction: async contracts between the application and exchange adapters.

Adapters translate between an exchange's own representation (responses, SDK
objects, status strings) and exchange-neutral domain types. Nothing
exchange-specific crosses the protocols in this package.
"""
