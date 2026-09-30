#!/usr/bin/env python3
"""Serve SAHVAI Web on an internal network with the headers it needs.

    python3 serve.py                 # http://localhost:8000/
    python3 serve.py --host 0.0.0.0 --port 8080

Sends Cross-Origin-Opener-Policy / Cross-Origin-Embedder-Policy so the page is
cross-origin isolated (multi-threaded CPU inference) without relying on the
service worker, and correct MIME types for .mjs / .wasm / .onnx. Standard library
only; no internet access is needed. For production, put the same headers on your
regular web server (see DEPLOY.md) and serve over HTTPS.
"""
import argparse, http.server, os, functools

class Handler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {**http.server.SimpleHTTPRequestHandler.extensions_map,
                      '.mjs': 'text/javascript', '.js': 'text/javascript', '.wasm': 'application/wasm',
                      '.onnx': 'application/octet-stream', '.nrrd': 'application/octet-stream'}

    def end_headers(self):
        self.send_header('Cross-Origin-Opener-Policy', 'same-origin')
        self.send_header('Cross-Origin-Embedder-Policy', 'require-corp')
        self.send_header('Cross-Origin-Resource-Policy', 'same-origin')
        self.send_header('X-Content-Type-Options', 'nosniff')
        super().end_headers()

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--host', default='127.0.0.1'); ap.add_argument('--port', type=int, default=8000)
    a = ap.parse_args()
    root = os.path.dirname(os.path.abspath(__file__))
    srv = http.server.ThreadingHTTPServer((a.host, a.port), functools.partial(Handler, directory=root))
    print(f'SAHVAI Web on http://{"localhost" if a.host in ("127.0.0.1", "0.0.0.0") else a.host}:{a.port}/  (Ctrl+C to stop)')
    srv.serve_forever()
