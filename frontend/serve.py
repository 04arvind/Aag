"""
Simple HTTP server to serve the frontend files.
Run this in the frontend directory: python serve.py
"""
import http.server
import socketserver
import webbrowser
import os

PORT = 8001

os.chdir(os.path.dirname(os.path.abspath(__file__)))

class Handler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()
    
    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

print(f"Serving frontend at http://localhost:{PORT}")
print("Opening browser...")
webbrowser.open(f"http://localhost:{PORT}/index.html")

with socketserver.TCPServer(("", PORT), Handler) as httpd:
    httpd.serve_forever()
