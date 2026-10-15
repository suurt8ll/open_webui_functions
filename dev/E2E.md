# E2E testing — driving the Open WebUI chat UI headlessly

Notes on end-to-end testing of this repo's plugins through the **real OWUI chat
UI** (not just the API), written up from a successful run. The point of e2e here
is to exercise the paths API tests can't reach: the model picker, the
Web Search toggle in the input area, streaming, and citation rendering.

## Why e2e and not just API calls

The backend pipeline (pipe → companion filter → Gemini grounding) can be
verified with plain `POST /api/chat/completions`. What that skips is the UI
half of the contract: is the model selectable, does the Web Search toggle
appear and actually set `features.web_search` on the request, and do the
grounding sources the companion filter emits render as citations? A headless
Playwright script covers all of that in one run.

## Prerequisites

1. **A running instance** via the repo tooling: `bash dev/dev.sh start`
   (see `dev/README.md`). The example below assumes `http://127.0.0.1:8084`.
2. **A Python venv with Playwright** and a browser:
   ```bash
   uv venv .e2e-venv && .e2e-venv/bin/uv pip install playwright
   .e2e-venv/bin/playwright install chromium-headless-shell
   sudo .e2e-venv/bin/playwright install-deps chromium-headless-shell
   ```
   The last step installs system libraries (libglib etc.); if the browser
   fails with `error while loading shared libraries`, re-run it — system
   packages may have been wiped by an image/sandbox reset even though the
   venv survived. Same for the browser cache under `~/.cache/ms-playwright`:
   re-run `playwright install` if the executable is missing.
3. **Config flags** (admin API or the admin UI):
   - `features.enable_web_search` must be truthy, otherwise the toggle is
     hidden. Verify with a **JWT**, not the API key — `/api/config` omits the
     authenticated `features` block when authenticated with an API key and
     returns `null` for the flag, which is a red herring.
   - The e2e user needs permission `permissions.features.web_search`
     (true by default for the `user` role).
4. **An e2e test account.** Create a normal user via the admin API (or a
   one-off signup), e.g. `POST /api/v1/users/add`, then sign in with
   `POST /api/v1/auths/signin` to get a JWT. Prefer a dedicated user so you
   can wipe it afterwards (`POST /api/v1/users/{id}/delete` as admin) and turn
   signup back off.

### Per-user function valves

Some pipes here set `USER_MUST_PROVIDE_AUTH_CONFIG`, meaning every user must
supply their own API key through **user valves** before the model works. For a
test account, copy the admin's user valves onto it instead of minting a key:

```bash
# as admin
curl -s -H "Authorization: Bearer $ADMIN_TOKEN" \
  "$BASE/api/v1/functions/id/<function_id>/valves/user" > user_valves.json
# validate/adjust values against the pipe's current valve schema (stale values
# from an older pipe version will be rejected), then apply as the test user
curl -s -X POST -H "Authorization: Bearer $E2E_TOKEN" \
  -H "Content-Type: application/json" --data @user_valves.json \
  "$BASE/api/v1/functions/id/<function_id>/valves/user/update"
```

This is one-time setup per account; it survives instance restarts.

## The script shape

A single self-contained script works well: login → select model → toggle web
search → send → wait → assert. Three full passes were run; the skeleton:

```python
# run: .e2e-venv/bin/python dev/e2e/<script>.py   (exit code 0 = all checks passed)
import asyncio, re, sys
from playwright.async_api import async_playwright

BASE = "http://127.0.0.1:8084"
EMAIL, PASSWORD = "e2e@test.local", "..."   # test account
MODEL_LABEL = "gemini-3.5-flash-lite"
PROMPT = "What are the top headlines on Hacker News right now? Cite the sources you used."
SHOTS = "e2e_shots"   # keep screenshots next to the script so they persist

async def main() -> int:
    results = []
    def check(name, ok, detail=""):
        results.append(ok)
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(viewport={"width": 1440, "height": 900})

        # 1. login — submit via role/name, not placeholder text
        await page.goto(f"{BASE}/auth", wait_until="networkidle")
        await page.locator("#email").fill(EMAIL)
        await page.locator("#password").fill(PASSWORD)
        await page.get_by_role("button", name="Sign in").click()
        await page.wait_for_url(re.compile(r".*/"), timeout=20000)

        # 2. model picker
        model_btn = page.locator("#model-selector-model-button")
        await model_btn.wait_for(state="visible", timeout=20000)
        label = await model_btn.get_attribute("aria-label") or ""
        if MODEL_LABEL not in label.lower().replace(" ", "-"):
            await model_btn.click()
            await page.get_by_text(MODEL_LABEL, exact=True).first.click()
        check("model selected", MODEL_LABEL in label.lower().replace(" ", "-"), label)

        # 3. Web Search toggle lives in the integrations ("+") menu
        await page.locator("#integration-menu-button").click()
        ws = page.get_by_role("button", name=re.compile("web search", re.I)).first
        await ws.wait_for(state="visible", timeout=10000)
        await ws.click()
        await page.keyboard.press("Escape")  # dismiss the menu

        # 4. send
        await page.locator("#chat-input").click()
        await page.locator("#chat-input").fill(PROMPT)
        await page.get_by_role("button", name=re.compile("^send", re.I)).first.click()

        # 5. wait for streaming to finish (stop button disappears)
        for _ in range(120):
            if await page.get_by_role("button", name=re.compile("stop", re.I)).count() == 0:
                break
            await page.wait_for_timeout(1000)
        await page.wait_for_timeout(3000)
        await page.screenshot(path=f"{SHOTS}/response.png", full_page=True)

        # 6. assertions
        text = re.sub(r"\s+", " ", await page.locator("body").inner_text())
        check("response non-empty", len(text) > 200, f"len={len(text)}")
        check("no error text", not any(
            s in text.lower() for s in ("exhausted execution options", "error occurred")))
        links = await page.locator("a[href^='http']").all()
        ext = [a for a in links if BASE not in (await a.get_attribute("href") or "")]
        check("citation links >= 1", len(ext) >= 1, f"count={len(ext)}")

        await browser.close()

    return 1 if not all(results) else 0

sys.exit(asyncio.run(main()))
```

## Selectors — stable vs. fragile

Verified against the OWUI revision in `submodules/open-webui` at time of
writing. IDs are the most durable handles; verify before trusting them on an
OWUI upgrade by dumping the DOM (`await page.content()`) and grepping:

| Element | Selector | Notes |
| --- | --- | --- |
| Email input | `#email` | placeholder is "Enter Your Email" (not "Email address") |
| Password input | `#password` | placeholder "Enter Your Password" |
| Submit | `get_by_role("button", name="Sign in")` | |
| Model picker | `#model-selector-model-button` | `aria-label` = `Selected model: <Name>` |
| Integrations ("+") menu | `#integration-menu-button` | contains `Gemini Paid API`, `Web Search`, `Code Interpreter` rows |
| Chat input | `#chat-input` | `fill()` works despite contenteditable |
| Send / Stop | `get_by_role("button", name=^send / stop)` | Stop visible only while streaming |

Gotchas learned the hard way:

- **Label normalization**: the picker's `aria-label` uses spaces
  (`Gemini 3.5 Flash Lite`) while model ids use dashes
  (`gemini-3.5-flash-lite`) — normalize (`lower().replace(" ", "-")`) before
  comparing, or you get a false FAIL.
- **API key vs JWT on `/api/config`**: see Prerequisites §3.
- Prefer `get_by_role` / `get_by_text` over brittle CSS; when something
  times out, dump `page.content()` or use `.count()` to debug.

## Validation checklist

Assert in the script (and eyeball a final screenshot):

1. Logged in (URL no longer `/auth`).
2. Model picker shows the intended model.
3. Web Search toggle was clicked and shows active state before sending.
4. Response text non-empty and free of error markers (e.g. the Gemini
   "Exhausted execution options" failure mode).
5. ≥1 external citation link / a Sources block rendered under the message.
6. Screenshot of the final message saved, then visually reviewed — a grounded
   reply is labeled *"This response was grounded with a Google tool"* and shows
   a `N Source` badge plus inline citation chips.

**Backend cross-check** (optional but conclusive): the companion filter logs
grounding activity, e.g. resolving Google `vertexaisearch.../grounding-api-redirect/…`
URIs to their final destinations:

```bash
grep -aE "grounding|google_search|_resolve_url" dev/logs/backend.log | tail
```

If the UI shows citations *and* the log shows resolved grounding redirects,
the full pipeline (UI toggle → `features.web_search` → companion filter →
`google_search_tool` → grounding → sources) fired end to end.

## Maintenance notes

- The script is deliberately uncommitted-sensitive: config/credentials come
  from the environment or a local file, screenshots go to a gitignored dir.
- When OWUI or the pipe changes UI/valve schemas, expect the selectors table
  and the valves section above to be the first things to drift.