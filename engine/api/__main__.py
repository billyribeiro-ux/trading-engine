"""
Run the dashboard API:  python -m engine.api  (or the `engine-api` script).

Honors HOST/PORT env vars. Loads .env-style FMP_API_KEY from the environment;
the server refuses /screen and /capabilities with 503 if the key is absent.
"""

from __future__ import annotations

import os


def main(argv: list[str] | None = None) -> int:
    import uvicorn

    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("engine.api.app:app", host=host, port=port, reload=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
