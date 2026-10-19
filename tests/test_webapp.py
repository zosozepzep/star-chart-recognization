import json
from pathlib import Path
import threading
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from src.webapp import Console, Handler, ThreadingHTTPServer, inside


@pytest.fixture
def server(tmp_path):
    console = Console(tmp_path, runtime='local')
    http = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    http.console = console
    thread = threading.Thread(target=http.serve_forever, daemon=True)
    thread.start()
    yield console, f'http://127.0.0.1:{http.server_port}'
    http.shutdown()
    http.server_close()
    thread.join()


def test_http_origin_token_and_path_boundaries(server):
    console, base = server
    boot = json.load(urlopen(base+'/api/bootstrap'))
    assert boot['token'] == console.token
    with pytest.raises(HTTPError) as error:
        urlopen(Request(base+'/api/start', data=b'{"kind":"benchmark"}', headers={'Content-Type':'application/json'}))
    assert error.value.code == 403
    with pytest.raises(HTTPError) as error:
        urlopen(Request(base+'/api/start', data=b'{}', headers={'X-Console-Token':console.token, 'Origin':'https://untrusted.invalid'}))
    assert error.value.code == 403
    with pytest.raises(HTTPError) as error:
        urlopen(Request(base+'/api/bootstrap', headers={'Host':'untrusted.invalid'}))
    assert error.value.code == 403
    with pytest.raises(ValueError):
        console.artifact('../../', 'report.json')
    with pytest.raises(ValueError):
        console.artifact('test', '../../README.md')
    with pytest.raises(ValueError):
        inside(console.root, '../outside')


def test_result_does_not_mutate_report_and_hides_failed_runs(server):
    console, base = server
    folder = console.root/'output'/'run-one'
    folder.mkdir(parents=True)
    report = dict(star_catalog={'sources':[dict(x=1., y=2., confirmed=True),dict(x=3., y=4., confirmed=False)]})
    text = json.dumps(report)
    (folder/'report.json').write_text(text, encoding='utf-8')
    (folder/'targets.csv').write_text('target_id,frame,time_utc,x_det,y_det,aperture_flux_adu_per_s\n1,2,2026-01-01T00:00:00,3,4,12\n', encoding='utf-8')
    result = json.load(urlopen(base+'/api/result?id=run-one'))
    assert result['stars'] == [[1.,2.]]
    assert result['tracks']['1'][0]['frame'] == 2
    assert (folder/'report.json').read_text(encoding='utf-8') == text
    (folder/'job.json').write_text(json.dumps({'state':'failed'}), encoding='utf-8')
    assert console.runs() == []


def test_worker_failure_is_reported_and_concurrent_start_is_rejected(tmp_path, monkeypatch):
    console = Console(tmp_path, runtime='local')
    def unavailable(*args, **kwargs):
        time.sleep(.1)
        raise OSError('worker deliberately unavailable')
    monkeypatch.setattr('src.webapp.subprocess.Popen', unavailable)
    started = console.start({'kind':'benchmark'})
    assert started['state'] == 'running'
    with pytest.raises(ValueError, match='已有任务'):
        console.start({'kind':'benchmark'})
    for _ in range(100):
        if console.status()['state'] != 'running': break
        time.sleep(.01)
    result = console.status()
    assert result['state'] == 'failed'
    assert 'worker deliberately unavailable' in result['logs']
    assert result['elapsed_seconds'] > 0
    receipt = tmp_path/'output'/started['run_id']/'job.json'
    assert json.loads(receipt.read_text(encoding='utf-8'))['state'] == 'failed'


def test_invalid_frame_cannot_launch_worker(tmp_path):
    console = Console(tmp_path)
    with pytest.raises(ValueError, match='有效的数据集'):
        console.start({'dataset':'data/missing','reference_frame':7})
    assert console.status() is None


def test_worker_starts_with_empty_output_and_persists_success_receipt(tmp_path, monkeypatch):
    """CLI rejects any nonempty output, including the console's own logs."""
    console = Console(tmp_path, runtime='local')
    from io import StringIO
    class Process:
        def __init__(self, command, **kwargs):
            output = tmp_path/command[command.index('--output')+1]
            assert not output.exists()
            output.mkdir(parents=True)
            (output/'benchmark.json').write_text('{}', encoding='utf-8')
            self.stdout = StringIO('finished\n')
        def wait(self): return 0
    monkeypatch.setattr('src.webapp.subprocess.Popen', Process)
    started = console.start({'kind':'benchmark'})
    for _ in range(100):
        if console.status()['state'] != 'running': break
        time.sleep(.01)
    assert console.status()['state'] == 'succeeded'
    folder = tmp_path/'output'/started['run_id']
    assert (folder/'run.log').read_text(encoding='utf-8') == 'finished\n'
    assert console.runs()[0]['id'] == started['run_id']
