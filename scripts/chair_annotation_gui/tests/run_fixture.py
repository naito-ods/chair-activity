"""Start an isolated synthetic-data instance for browser tests; no real annotations."""
import argparse
import secrets
import sys
from http.server import ThreadingHTTPServer
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from server import Store, handler
from test_server import fixture

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=Path, required=True)
parser.add_argument('--port', type=int, default=8766)
args = parser.parse_args()
fixture(args.root)
store = Store(args.root, args.root / 'Output/annotations')
server = ThreadingHTTPServer(('127.0.0.1', args.port), handler(store, secrets.token_urlsafe(32), args.port))
print(f'Test fixture http://127.0.0.1:{args.port}', flush=True)
server.serve_forever()
