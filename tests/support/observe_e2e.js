/* Real Edge E2E for observe: layout, clicks, focus and native EventSource reconnect.
 * Opt-in through tests/shared/test_observe_browser_e2e.py. No Safari or paid/GPU path.
 */
const fs = require('fs');
const os = require('os');
const path = require('path');
const http = require('http');
const { once } = require('events');
const { spawn } = require('child_process');
const { chromium } = require(process.env.PLAYWRIGHT_CORE_PATH || 'playwright-core');

function arg(name, fallback = '') {
  const index = process.argv.indexOf(name);
  return index >= 0 && process.argv[index + 1] ? process.argv[index + 1] : fallback;
}
function sleep(ms) { return new Promise(resolve => setTimeout(resolve, ms)); }
function assert(condition, message) { if (!condition) throw new Error(message); }
function request(url, method = 'GET', timeout = 1000) {
  return new Promise((resolve, reject) => {
    const req = http.request(url, { method, timeout }, res => {
      let body = '';
      res.on('data', chunk => { body += chunk; });
      res.on('end', () => resolve({ status: res.statusCode, body }));
    });
    req.on('error', reject);
    req.on('timeout', () => req.destroy(new Error('HTTP timeout')));
    req.end();
  });
}
async function waitHttp(port, timeout = 30000) {
  const deadline = Date.now() + timeout;
  while (Date.now() < deadline) {
    try {
      const result = await request(`http://127.0.0.1:${port}/api/v2/overview`);
      if (result.status === 200) return;
    } catch (_) { /* starting */ }
    await sleep(100);
  }
  throw new Error(`observe server did not become ready on ${port}`);
}
function writeRun(root, id, spent) {
  const dir = path.join(root, id);
  fs.mkdirSync(dir, { recursive: true });
  const now = Date.now() / 1000;
  const state = {
    pid: 1, started_at: now, updated_at: now, phase: 'done',
    task: 'relu', dtype: 'fp16', size: 'small', stop_reason: 'max_rounds',
    spent_usd: spent, budget_usd: 1.0, best_speedup: 0.9, trace: [],
  };
  const file = path.join(dir, 'state.json');
  fs.writeFileSync(file, JSON.stringify(state));
  return file;
}
function publish(file, spent) {
  const doc = JSON.parse(fs.readFileSync(file, 'utf8'));
  doc.spent_usd = spent;
  doc.updated_at = Date.now() / 1000;
  const temporary = `${file}.e2e-${process.pid}`;
  fs.writeFileSync(temporary, JSON.stringify(doc));
  fs.renameSync(temporary, file);
}
async function stopServer(child) {
  if (!child || child.exitCode !== null || child.signalCode) return;
  try { process.kill(-child.pid, 'SIGTERM'); } catch (_) { return; }
  const exited = await Promise.race([once(child, 'exit').then(() => true), sleep(10000).then(() => false)]);
  if (!exited) {
    try { process.kill(-child.pid, 'SIGKILL'); } catch (_) { /* already gone */ }
    await Promise.race([once(child, 'exit'), sleep(2000)]);
  }
}

async function main() {
  const repo = path.resolve(arg('--repo', process.cwd()));
  const out = path.resolve(arg('--out', ''));
  if (!out) throw new Error('--out is required');
  fs.mkdirSync(out, { recursive: true });
  const temporary = fs.mkdtempSync(path.join(os.tmpdir(), 'shapeforge-browser-e2e-'));
  const runs = path.join(temporary, 'runs');
  const firstState = writeRun(runs, 'e2e/one', 0.2);
  writeRun(runs, 'e2e/two', 0.4);
  const serverLog = path.join(out, 'server.log');
  const logFd = fs.openSync(serverLog, 'a');
  const port = 18000 + (process.pid % 10000);
  const env = {
    ...process.env,
    AI_LOWERLING_RUNS_DIR: runs,
    AI_LOWERLING_OBSERVE_DIR: path.join(temporary, 'cache'),
    AI_LOWERLING_BOOKS: path.join(temporary, 'books.sqlite'),
    OBSERVE_DB: path.join(temporary, 'observe.sqlite'),
    OBSERVE_CARRYOVER: path.join(temporary, 'ledger_carryover.json'),
    TAIC_CONFIG: path.join(temporary, 'taic.toml'),
    OBSERVE_POLL_S: '0.05', OBSERVE_TICK_INTERVAL: '0', OBSERVE_FULL_TICK_INTERVAL: '0',
    OBSERVE_REFRESH_ON_READ: '1', OBSERVE_CONTROL_DRYRUN: '1', OBSERVE_PORT: String(port),
  };
  let server = null;
  let browser = null;
  let page = null;
  const evidence = {
    schema_id: 'local.observe-browser-e2e.1', status: 'running',
    browser: 'Microsoft Edge (real Chromium engine)', headless: true, safari_used: false,
    read_only_http_guard: true, mutating_controls_clicked: false,
    paid_model_calls: 0, gpu_calls: 0, publication_allowed: false,
    layout_widths: [], clicks: [], focus: {}, reconnect: {},
  };
  function startServer() {
    const child = spawn('uv', ['run', 'python', '-m', 'tests.support.observe_e2e_server'], {
      cwd: repo, env, detached: true, stdio: ['ignore', logFd, logFd],
    });
    return child;
  }
  try {
    server = startServer();
    await waitHttp(port);
    for (const method of ['HEAD', 'POST']) {
      for (const route of ['/api/budget/reset', '/api/stop']) {
        const result = await request(`http://127.0.0.1:${port}${route}`, method);
        assert(result.status === 403, `${method} ${route} escaped the read-only guard: ${result.status}`);
      }
    }
    evidence.mutation_negative_checks = ['HEAD reset', 'HEAD stop', 'POST reset', 'POST stop'];
    browser = await chromium.launch({ channel: 'msedge', headless: true });
    evidence.browser_version = browser.version();
    const context = await browser.newContext({ viewport: { width: 1280, height: 900 } });
    page = await context.newPage();
    const consoleErrors = [];
    const pageErrors = [];
    const detailRequests = [];
    const navigations = [];
    page.on('console', message => {
      if (message.type() === 'error') {
        consoleErrors.push({ text: message.text(), url: message.location().url, restartWindow });
      }
    });
    page.on('pageerror', error => pageErrors.push(String(error)));
    page.on('dialog', dialog => dialog.dismiss());
    page.on('request', request => {
      const pathname = new URL(request.url()).pathname.toLowerCase();
      if (/\/api\/v2\/runs\/e2e(?:\/|%2f)one$/i.test(pathname)) {
        detailRequests.push(request.url());
      }
    });
    page.on('framenavigated', frame => { if (frame === page.mainFrame()) navigations.push(frame.url()); });
    await page.addInitScript(() => {
      const Native = window.EventSource;
      window.__e2eEvents = { constructors: 0, opens: 0, errors: 0, messages: [], sources: [] };
      function ObservedEventSource(...args) {
        const source = new Native(...args);
        const events = window.__e2eEvents;
        events.constructors += 1;
        events.sources.push(source);
        source.addEventListener('open', () => { events.opens += 1; });
        source.addEventListener('error', () => { events.errors += 1; });
        source.addEventListener('message', event => {
          try {
            const frame = JSON.parse(event.data);
            events.messages.push({ revision: frame.revision || frame.index_revision || 0, runs: frame.runs || [] });
          } catch (_) { /* observer only */ }
        });
        return source;
      }
      ObservedEventSource.prototype = Native.prototype;
      window.EventSource = ObservedEventSource;
    });
    const base = `http://127.0.0.1:${port}`;
    let restartWindow = false;
    await page.goto(base, { waitUntil: 'domcontentloaded' });
    await page.waitForSelector('#runs tbody tr[data-run="e2e/one"]', { timeout: 30000 });
    await page.locator('#runs tbody tr[data-run="e2e/one"] td:first-child').click();
    await page.waitForFunction(() => new URL(location.href).searchParams.get('run') === 'e2e/one');
    await page.waitForFunction(() => document.querySelector('#detail-task').textContent.includes('relu'), undefined, { timeout: 10000 });
    assert(await page.locator('#view-detail.on').count() === 1, 'detail view is not visible after run click');
    evidence.clicks.push('run row opens detail and updates URL');

    for (const [id, panel] of [['runs-toggle', 'runs-panel'], ['detail-toggle', 'detail-panel'], ['side-toggle', 'workspace']]) {
      await page.locator(`#${id}`).click();
      assert(await page.locator(`#${panel}.collapsed`).count() === 1, `${id} did not collapse ${panel}`);
      assert((await page.locator(`#${id}`).getAttribute('aria-expanded')) === 'false', `${id} aria-expanded did not close`);
      await page.locator(`#${id}`).click();
      evidence.clicks.push(`${id} collapses and restores ${panel}`);
    }

    await page.locator('#search-open').click();
    await page.waitForSelector('#search-modal.open');
    assert(await page.evaluate(() => document.activeElement.id) === 'search-input', 'search did not focus input');
    await page.keyboard.press('Shift+Tab');
    assert(await page.evaluate(() => document.activeElement.id) === 'search-close', 'Shift+Tab did not wrap to close');
    await page.keyboard.press('Tab');
    assert(await page.evaluate(() => document.activeElement.id) === 'search-input', 'Tab did not wrap to input');
    await page.keyboard.press('Escape');
    assert(await page.locator('#search-modal.open').count() === 0, 'Escape did not close search');
    assert(await page.evaluate(() => document.activeElement.id) === 'search-open', 'focus did not return to opener');
    evidence.focus.modal = 'input focus, two-sided Tab trap, Escape and opener restoration pass';

    const runRows = page.locator('#runs tbody tr[data-run]');
    assert(await runRows.count() === 2, 'expected two real run rows');
    assert(await page.locator('#runs tbody tr[data-run][tabindex="0"]').count() === 1, 'roving tabindex needs exactly one stop');
    await page.locator('#runs tbody tr[data-run="e2e/two"]').focus();
    await page.keyboard.press('ArrowDown');
    assert(await page.evaluate(() => document.activeElement.dataset.run) === 'e2e/one', 'ArrowDown did not move actual focus');
    const outline = await page.evaluate(() => {
      const style = getComputedStyle(document.activeElement);
      return { style: style.outlineStyle, width: style.outlineWidth };
    });
    assert(outline.style !== 'none' && parseFloat(outline.width) > 0, 'focused run has no visible outline');
    evidence.focus.roving = outline;
    await page.locator('#runs tbody tr[data-run="e2e/one"] td:first-child').click();

    for (const width of [1280, 760, 620, 480, 320]) {
      await page.setViewportSize({ width, height: 900 });
      await sleep(100);
      const layout = await page.evaluate(() => {
        const header = document.querySelector('header');
        const main = document.querySelector('main');
        const split = document.querySelector('.detail-split');
        const detail = document.querySelector('.detail-main');
        const headerRect = header ? header.getBoundingClientRect() : null;
        const mainRect = main ? main.getBoundingClientRect() : null;
        return {
          viewport: innerWidth, scrollWidth: document.documentElement.scrollWidth,
          headerMainOverlap: Boolean(headerRect && mainRect && mainRect.top + 1 < headerRect.bottom),
          detailWidthRatio: split && detail ? detail.getBoundingClientRect().width / split.getBoundingClientRect().width : 1,
        };
      });
      assert(layout.scrollWidth <= width, `horizontal overflow at ${width}: ${layout.scrollWidth}`);
      assert(!layout.headerMainOverlap, `header/main overlap at ${width}`);
      assert(Math.abs(layout.detailWidthRatio - 1) < 0.02, `detail does not occupy full split width at ${width}`);
      await page.screenshot({ path: path.join(out, `layout-${width}.png`), fullPage: true });
      evidence.layout_widths.push(width);
    }
    await page.setViewportSize({ width: 1280, height: 900 });
    await page.locator('#runs tbody tr[data-run="e2e/one"] td:first-child').click();
    await sleep(300);

    const beforeRestartRequests = detailRequests.length;
    const beforeRestartNavigations = navigations.length;
    const beforeUrl = page.url();
    const beforeRestartRevision = await page.evaluate(() => {
      const messages = window.__e2eEvents.messages;
      return messages.length ? messages[messages.length - 1].revision : 0;
    });
    restartWindow = true;
    await stopServer(server);
    server = null;
    await page.waitForFunction(() => window.__e2eEvents.errors >= 1, undefined, { timeout: 15000 });
    await page.waitForFunction(() => window.__e2eEvents.sources[0].readyState === 0, undefined, { timeout: 15000 });
    server = startServer();
    await waitHttp(port);
    await page.waitForFunction(() => window.__e2eEvents.opens >= 2 && window.__e2eEvents.sources[0].readyState === 1, undefined, { timeout: 30000 });
    restartWindow = false;
    await sleep(1000);
    assert(page.url() === beforeUrl, 'browser navigated during reconnect');
    assert(navigations.length === beforeRestartNavigations, 'reconnect added a main-frame navigation');
    assert(detailRequests.length === beforeRestartRequests, 'unchanged revision caused duplicate detail fetch after reconnect');
    const reconnectRevision = await page.evaluate(() => {
      const messages = window.__e2eEvents.messages;
      return messages.length ? messages[messages.length - 1].revision : 0;
    });
    assert(reconnectRevision === beforeRestartRevision, 'restart changed the unchanged snapshot revision');

    const beforePublishMessages = await page.evaluate(() => window.__e2eEvents.messages.length);
    publish(firstState, 0.35);
    await page.waitForFunction(() => document.querySelector('#detail-budget').textContent.includes('0.35'), undefined, { timeout: 15000 });
    await sleep(500);
    const changedFrames = await page.evaluate(start => window.__e2eEvents.messages.slice(start).filter(frame => frame.runs.includes('e2e/one') && frame.revision > 0), beforePublishMessages);
    assert(changedFrames.length === 1, `expected one changed frame, got ${changedFrames.length}`);
    assert(detailRequests.length === beforeRestartRequests + 1, `changed revision should fetch detail exactly once: ${detailRequests.length - beforeRestartRequests}`);
    assert(await page.locator('#runs tbody tr[data-run="e2e/one"]').count() === 1, 'run row duplicated after reconnect/update');
    const eventCounts = await page.evaluate(() => ({
      constructors: window.__e2eEvents.constructors,
      opens: window.__e2eEvents.opens,
      errors: window.__e2eEvents.errors,
      readyState: window.__e2eEvents.sources[0].readyState,
    }));
    assert(eventCounts.constructors === 1 && eventCounts.opens >= 2 && eventCounts.errors >= 1 && eventCounts.readyState === 1,
      `native EventSource reconnect failed: ${JSON.stringify(eventCounts)}`);
    evidence.reconnect = {
      real_server_restart: true, unchanged_revision_no_detail_refetch: true,
      changed_revision_frames: changedFrames.length, changed_revision_detail_fetches: 1,
      run_rows_after_update: 1, url_unchanged: true, ...eventCounts,
    };
    const expectedNetworkErrors = [];
    const unexpectedConsoleErrors = [];
    for (const error of consoleErrors) {
      if (error.restartWindow && error.url === `${base}/api/v2/stream` && error.text === 'Failed to load resource: net::ERR_INCOMPLETE_CHUNKED_ENCODING') {
        expectedNetworkErrors.push({ ...error, reason: 'intentional real server restart / SSE EOF' });
      } else if (error.url.endsWith('/favicon.ico') && error.text.includes('404')) {
        expectedNetworkErrors.push({ ...error, reason: 'the app has no favicon route; unrelated to the tested workflow' });
      } else {
        unexpectedConsoleErrors.push(error);
      }
    }
    const expectedDisconnectErrors = expectedNetworkErrors.filter(error => error.url === `${base}/api/v2/stream`);
    assert(expectedDisconnectErrors.length === 1, `expected exactly one intentional stream EOF, got ${expectedDisconnectErrors.length}`);
    evidence.expected_network_errors = expectedNetworkErrors;
    evidence.console_errors = unexpectedConsoleErrors;
    evidence.page_errors = pageErrors;
    assert(unexpectedConsoleErrors.length === 0, `console errors: ${JSON.stringify(unexpectedConsoleErrors)}`);
    assert(pageErrors.length === 0, `page errors: ${pageErrors.join(' | ')}`);
    evidence.navigation_count_before_restart = beforeRestartNavigations;
    evidence.navigation_count_after_reconnect = navigations.length;
    evidence.navigations = navigations;
    evidence.status = 'passed';
  } catch (error) {
    evidence.status = 'failed';
    evidence.error = String(error && error.stack || error);
    if (page) {
      try { await page.screenshot({ path: path.join(out, 'failure.png'), fullPage: true }); } catch (_) { /* closing */ }
    }
    throw error;
  } finally {
    evidence.navigations = evidence.navigations || [];
    fs.writeFileSync(path.join(out, 'evidence.json'), JSON.stringify(evidence, null, 2) + '\n');
    if (browser) await browser.close();
    await stopServer(server);
    fs.closeSync(logFd);
    fs.rmSync(temporary, { recursive: true, force: true });
  }
}

main().catch(error => { console.error(error); process.exit(1); });
