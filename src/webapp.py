"""Local offline demonstration console. Server uses only Python's standard library.

Run from the repository: python -m src.webapp
The default worker runs the existing CPU Docker image with network disabled.
"""
from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
import os
from pathlib import Path
import re
import secrets
import shutil
import subprocess
import sys
import threading
import time
from urllib.parse import parse_qs, urlparse
import uuid

ROOT = Path(__file__).resolve().parents[1]
STATIC = Path(__file__).with_name('web')


def inside(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('路径超出允许的目录')
    return path


def fits_stamp(path):
    """Read DATE-OBS for display ordering without loading scientific dependencies."""
    with path.open('rb') as fh:
        for _ in range(64):
            block = fh.read(2880)
            if not block:
                break
            for offset in range(0, len(block), 80):
                card = block[offset:offset+80].decode('ascii', errors='replace')
                if card.startswith('DATE-OBS='):
                    return card.split("'")[1]
                if card.startswith('END '):
                    return path.name
    return path.name


class Console:
    def __init__(self, root=ROOT, runtime='docker', image='star-chart:cpu'):
        self.root, self.runtime, self.image = Path(root).resolve(), runtime, image
        self.token = secrets.token_urlsafe(32)
        self.lock = threading.RLock()
        self.job = None
        self.process = None

    def datasets(self):
        data = self.root/'data'
        groups = {}
        for path in data.rglob('*'):
            if path.suffix.lower() in ('.fits', '.fit', '.fts') and path.is_file():
                if path.resolve().is_relative_to(data.resolve()):
                    groups.setdefault(path.parent, []).append(path)
        return [dict(id=folder.relative_to(self.root).as_posix(), name=folder.name,
                     frames=[p.name for p in sorted(paths, key=fits_stamp)])
                for folder, paths in sorted(groups.items())]

    def runs(self):
        result = []
        for pattern in ('report.json', 'benchmark.json'):
            for p in (self.root/'output').rglob(pattern):
                if not p.resolve().is_relative_to((self.root/'output').resolve()):
                    continue
                # Ignore intermediate reports until the worker has completed.
                with self.lock:
                    if self.job and self.job['state'] in ('running', 'stopping') and self.job['run_id'] == p.parent.relative_to(self.root/'output').as_posix():
                        continue
                receipt = p.parent/'job.json'
                pending = self.root/'output'/'.console-jobs'/p.parent.name/'job.json'
                if pending.exists():
                    receipt = pending
                if receipt.exists():
                    try:
                        if json.loads(receipt.read_text(encoding='utf-8'))['state'] != 'succeeded':
                            continue
                    except (ValueError, KeyError):
                        continue
                result.append(dict(id=p.parent.relative_to(self.root/'output').as_posix(),
                                   kind='benchmark' if pattern == 'benchmark.json' else 'analysis',
                                   modified=p.stat().st_mtime))
        return sorted(result, key=lambda r: r['modified'], reverse=True)[:80]

    def status(self):
        with self.lock:
            if self.job is None:
                return None
            value = {k: v for k, v in self.job.items() if k not in ('monotonic', 'logs')}
            value['logs'] = list(self.job['logs'])
            value['elapsed_seconds'] = self.job.get('elapsed_seconds', time.perf_counter()-self.job['monotonic'])
            return value

    def start(self, request):
        kind = request.get('kind', 'analysis')
        if kind not in ('analysis', 'benchmark'):
            raise ValueError('未知任务类型')
        with self.lock:
            if self.job and self.job['state'] in ('running', 'stopping'):
                raise ValueError('已有任务运行中，请等待完成或停止任务')
            args = []
            if kind == 'analysis':
                datasets = {d['id']: d for d in self.datasets()}
                dataset = datasets.get(request.get('dataset'))
                frame = request.get('reference_frame')
                if dataset is None or type(frame) is not int or not 0 <= frame < len(dataset['frames']):
                    raise ValueError('请选择有效的数据集和指定帧')
                args = ['-m', 'src.cli', '--input', dataset['id'], '--reference-frame', str(frame), '--viewer']
                if request.get('catalog'):
                    args += ['--catalog', 'docs/reference/rst19-gaia-dr3.csv']
            else:
                args = ['-m', 'src.validate.benchmark']
            identifier = f"console-{kind}-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
            output = self.root/'output'/identifier
            bookkeeping = self.root/'output'/'.console-jobs'/identifier
            bookkeeping.mkdir(parents=True)
            args += ['--output', 'output/'+identifier]
            container = 'star-chart-'+uuid.uuid4().hex[:12]
            if self.runtime == 'docker':
                command = ['docker', 'run', '--rm', '--pull=never', '--name', container, '--network', 'none',
                           '-v', f'{self.root}:/workspace', '-w', '/workspace',
                           '-e', 'PYTHONPATH=/workspace', '-e', 'MPLBACKEND=Agg',
                           '-e', 'PYTHONUNBUFFERED=1', '-e', 'PYTHONIOENCODING=utf-8',
                           self.image, 'python', *args]
            else:
                command = [sys.executable, '-u', *args]
            self.job = dict(run_id=identifier, kind=kind, state='running', logs=deque(maxlen=240),
                            started_utc=datetime.now(timezone.utc).isoformat(), monotonic=time.perf_counter(),
                            container=container, runtime=self.runtime, command=command)
            (bookkeeping/'job.json').write_text(json.dumps(self.status(), ensure_ascii=False, indent=2), encoding='utf-8')
            threading.Thread(target=self._work, args=(command, output), daemon=True).start()
            return self.status()

    def _work(self, command, output):
        code, error = -1, None
        bookkeeping = self.root/'output'/'.console-jobs'/output.name
        try:
            env = {**os.environ, 'PYTHONIOENCODING': 'utf-8', 'MPLBACKEND': 'Agg'}
            with (bookkeeping/'run.log').open('w', encoding='utf-8') as log:
                with self.lock:
                    if self.job['state'] == 'stopping':
                        raise RuntimeError('任务已停止')
                    self.process = subprocess.Popen(command, cwd=self.root, env=env,
                                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                                    text=True, encoding='utf-8', errors='replace',
                                                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
                    process = self.process
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    with self.lock:
                        self.job['logs'].append(line.rstrip())
                code = process.wait()
        except (OSError, RuntimeError) as exc:
            error = str(exc)
        finally:
            with self.lock:
                stopped = self.job['state'] == 'stopping'
                self.job['state'] = 'cancelled' if stopped else ('succeeded' if code == 0 else 'failed')
                self.job['exit_code'] = code
                self.job['elapsed_seconds'] = time.perf_counter()-self.job['monotonic']
                if error:
                    self.job['logs'].append(error)
                self.process = None
                output.mkdir(parents=True, exist_ok=True)
                receipt = json.dumps(self.status(), ensure_ascii=False, indent=2)
                (output/'job.json').write_text(receipt, encoding='utf-8')
                (bookkeeping/'job.json').write_text(receipt, encoding='utf-8')
                if (bookkeeping/'run.log').exists():
                    shutil.copyfile(bookkeeping/'run.log', output/'run.log')

    def stop(self):
        with self.lock:
            if not self.job or self.job['state'] != 'running':
                raise ValueError('没有正在运行的任务')
            self.job['state'] = 'stopping'
            process, container = self.process, self.job['container']
        if process and process.poll() is None:
            if self.runtime == 'docker':
                subprocess.run(['docker', 'stop', '--time', '2', container], capture_output=True, timeout=15)
            else:
                process.terminate()
        return self.status()

    def artifact(self, run, name):
        if not re.fullmatch(r'(report\.json|benchmark\.json|stars\.csv|targets\.csv|job\.json|run\.log|faintest-star\.png|motion-tracks\.png|reference-preview\.png|target-\d+-f\d+\.png)', name):
            raise ValueError('文件不在可导出清单内')
        folder = inside(self.root/'output', run)
        return inside(folder, name)

    def result(self, identifier):
        folder = inside(self.root/'output', identifier)
        if (folder/'benchmark.json').is_file():
            return dict(kind='benchmark', report=json.loads(inside(folder, 'benchmark.json').read_text(encoding='utf-8')))
        report = json.loads(inside(folder, 'report.json').read_text(encoding='utf-8'))
        sources = report['star_catalog'].pop('sources', [])
        import csv
        tracks = {}
        with inside(folder, 'targets.csv').open(encoding='utf-8-sig', newline='') as fh:
            for row in csv.DictReader(fh):
                tracks.setdefault(row['target_id'], []).append(dict(frame=int(row['frame']),
                    x=float(row['x_det']), y=float(row['y_det']), time_utc=row['time_utc'],
                    flux=float(row['aperture_flux_adu_per_s'])))
        receipt = json.loads(inside(folder, 'job.json').read_text(encoding='utf-8')) if (folder/'job.json').exists() else None
        return dict(kind='analysis', report=report, tracks=tracks,
                    stars=[[s['x'], s['y']] for s in sources if s['confirmed']], receipt=receipt)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def reply(self, value, status=200):
        body = json.dumps(value, ensure_ascii=False, allow_nan=False).encode('utf-8')
        self.send_body(body, 'application/json; charset=utf-8', status)

    def send_body(self, body, content_type, status=200, attachment=None):
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Content-Security-Policy', "default-src 'self'; img-src 'self' data:; style-src 'self'; script-src 'self'; connect-src 'self'; frame-ancestors 'none'")
        if attachment:
            self.send_header('Content-Disposition', f'attachment; filename="{attachment}"')
        self.end_headers()
        self.wfile.write(body)

    def same_host(self):
        host = self.headers.get('Host', '')
        return host in (f'127.0.0.1:{self.server.server_port}', f'localhost:{self.server.server_port}')

    def do_GET(self):
        if not self.same_host():
            return self.reply({'error': '仅允许本机访问'}, 403)
        url = urlparse(self.path)
        query = parse_qs(url.query)
        console = self.server.console
        try:
            if url.path == '/api/bootstrap':
                self.reply(dict(token=console.token, datasets=console.datasets(), runs=console.runs(),
                                runtime=console.runtime, image=console.image, job=console.status()))
            elif url.path == '/api/status':
                self.reply(dict(job=console.status()))
            elif url.path == '/api/runs':
                self.reply(console.runs())
            elif url.path == '/api/result':
                self.reply(console.result(query['id'][0]))
            elif url.path == '/api/artifact':
                name = query['name'][0]
                path = console.artifact(query['run'][0], name)
                content_type = mimetypes.guess_type(name)[0] or 'application/octet-stream'
                self.send_body(path.read_bytes(), content_type, attachment=name if 'download' in query else None)
            elif url.path in ('/', '/app.js', '/style.css'):
                path = STATIC / ('index.html' if url.path == '/' else url.path[1:])
                self.send_body(path.read_bytes(), (mimetypes.guess_type(path.name)[0] or 'text/plain')+'; charset=utf-8')
            else:
                self.reply({'error': '页面不存在'}, 404)
        except (ValueError, OSError, KeyError, IndexError) as exc:
            self.reply({'error': str(exc)}, 400)

    def do_POST(self):
        origin = self.headers.get('Origin')
        if (not self.same_host() or self.headers.get('X-Console-Token') != self.server.console.token or
            (origin and origin != 'http://'+self.headers.get('Host', ''))):
            return self.reply({'error': '请求来源校验失败'}, 403)
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= 8192:
                raise ValueError('请求体长度无效')
            request = json.loads(self.rfile.read(length))
            if not isinstance(request, dict):
                raise ValueError('请求必须是 JSON 对象')
            if self.path == '/api/start':
                self.reply(self.server.console.start(request), 202)
            elif self.path == '/api/stop':
                self.reply(self.server.console.stop())
            else:
                self.reply({'error': '操作不存在'}, 404)
        except (ValueError, OSError, subprocess.TimeoutExpired) as exc:
            self.reply({'error': str(exc)}, 400)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--runtime', choices=['docker', 'local'], default='docker')
    parser.add_argument('--image', default=os.environ.get('SC_IMAGE', 'star-chart:cpu'))
    args = parser.parse_args()
    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    server.console = Console(runtime=args.runtime, image=args.image)
    print(f'AstraVision: http://127.0.0.1:{server.server_port} | worker={args.runtime}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        if server.console.status() and server.console.status()['state'] == 'running':
            server.console.stop()
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
