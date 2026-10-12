"""Playwright end-to-end checks for responsive navigation on Streamlit."""
from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest
from playwright.sync_api import expect

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def app_url():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    env = dict(os.environ, STREAMLIT_SERVER_HEADLESS="true")
    process = subprocess.Popen(
        [
            sys.executable, "-m", "streamlit", "run", str(ROOT / "app.py"),
            f"--server.port={port}", "--server.address=127.0.0.1",
            "--browser.gatherUsageStats=false",
        ],
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    url = f"http://127.0.0.1:{port}"
    try:
        for _ in range(100):
            if process.poll() is not None:
                raise RuntimeError("Streamlit exited before becoming healthy")
            try:
                with urllib.request.urlopen(url + "/_stcore/health", timeout=1) as response:
                    if response.read().decode().strip() == "ok":
                        break
            except (OSError, urllib.error.URLError):
                time.sleep(.2)
        else:
            raise RuntimeError("Streamlit health endpoint never became available")
        yield url
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()


@pytest.mark.parametrize("width,height", [(360, 780), (390, 844), (768, 1024), (980, 760), (1440, 900)])
def test_hamburger_anchored_right_and_closes_after_navigation(page, app_url, width, height):
    page.set_viewport_size({"width": width, "height": height})
    page.goto(app_url)
    trigger = page.get_by_role("button", name="Open navigation menu")
    expect(trigger).to_be_visible(timeout=30000)
    coords = trigger.bounding_box()
    assert coords is not None
    # The control must remain within the browser viewport at all tested widths.
    assert 0 <= coords["x"]
    assert coords["x"] + coords["width"] <= width + 2

    trigger.click()
    option = page.get_by_role("link", name="Batch Screening")
    expect(option).to_be_visible()
    option.click()
    expect(page.get_by_text("Upload CSV, TXT or Excel file containing SMILES", exact=True)).to_be_visible(timeout=12000)
    expect(page.get_by_role("link", name="Batch Screening")).to_have_count(0)
    expect(page.get_by_role("button", name="Open navigation menu")).to_be_visible()


def test_menu_escape_and_outside_click(page, app_url):
    page.set_viewport_size({"width": 390, "height": 844})
    page.goto(app_url)
    page.get_by_role("button", name="Open navigation menu").click()
    expect(page.get_by_role("link", name="Model & Methods")).to_be_visible()
    page.keyboard.press("Escape")
    expect(page.get_by_role("link", name="Model & Methods")).to_have_count(0)

    page.get_by_role("button", name="Open navigation menu").click()
    page.get_by_text("Molecular editor", exact=True).click()
    expect(page.get_by_role("button", name="Model & Methods")).to_have_count(0)


@pytest.mark.parametrize("width,height", [(320, 700), (360, 780), (390, 844), (430, 932), (768, 1024)])
def test_molecular_drawer_fully_within_mobile_viewport(page, app_url, width, height):
    """JSME frame, not just its outer card, must fit the screen."""
    page.set_viewport_size({"width": width, "height": height})
    page.goto(app_url)
    editor_frame = page.locator('iframe').first
    expect(editor_frame).to_be_visible(timeout=30000)
    # Wait for the width-probe callback to determine actual column width.
    page.wait_for_timeout(700)
    dimensions = editor_frame.bounding_box()
    assert dimensions is not None
    assert dimensions["x"] >= 0, dimensions
    assert dimensions["x"] + dimensions["width"] <= width + 2, dimensions
    # No fixed 550px editor is allowed on a narrow phone.
    if width <= 430:
        assert dimensions["width"] <= width - 12, dimensions
    assert dimensions["height"] >= 320, dimensions


def test_drawer_keeps_full_width_on_mobile_landscape(page, app_url):
    page.set_viewport_size({"width": 844, "height": 390})
    page.goto(app_url)
    frame = page.locator("iframe").first
    expect(frame).to_be_visible(timeout=30000)
    bounds = frame.bounding_box()
    assert bounds is not None
    assert bounds["x"] >= 0
    assert bounds["x"] + bounds["width"] <= 846
    # On tablet-size / landscape viewports, the editor is stacked full-width.
    assert bounds["width"] >= 500


@pytest.mark.parametrize("width,height", [(320, 700), (360, 780), (390, 844), (430, 932)])
def test_mobile_view_not_cramped(page, app_url, width, height):
    """Phone screen reflows content rather than shrinking multi-column cards."""
    page.set_viewport_size({"width": width, "height": height})
    page.goto(app_url)
    expect(page.get_by_text("Molecular editor")).to_be_visible(timeout=30000)
    iframe = page.locator("iframe").first
    expect(iframe).to_be_visible(timeout=30000)
    # Frame dimensions must be bounded by the user's actual screen.
    bounds = iframe.bounding_box()
    assert bounds is not None
    assert bounds["x"] >= 0
    assert bounds["x"] + bounds["width"] <= width + 2
    assert bounds["width"] >= max(220, width - 65), bounds
    # No horizontal overflow caused by a forcibly shrunken desktop workspace.
    scroll = page.evaluate("() => ({scroll: document.documentElement.scrollWidth, view: document.documentElement.clientWidth})")
    assert scroll["scroll"] <= scroll["view"] + 3, scroll

    pic50 = page.locator('[data-testid="stMetric"]').filter(has_text="Predicted pIC₅₀").first
    ic50 = page.locator('[data-testid="stMetric"]').filter(has_text="Estimated IC₅₀").first
    expect(pic50).to_be_visible(timeout=30000)
    expect(ic50).to_be_visible()
    first = pic50.bounding_box()
    second = ic50.bounding_box()
    assert first is not None and second is not None
    assert second["y"] >= first["y"] + first["height"] - 2, (first, second)


def test_direct_batch_route(page, app_url):
    page.goto(app_url + "?view=batch")
    expect(page.get_by_text("Upload CSV, TXT or Excel file containing SMILES", exact=True)).to_be_visible(timeout=30000)


def test_menu_switches_repeatedly_without_page_reload(page, app_url):
    """Selecting a menu item must change the displayed Streamlit page each time."""
    page.set_viewport_size({"width": 390, "height": 844})
    page.goto(app_url)
    expect(page.get_by_text("Molecular editor", exact=True)).to_be_visible(timeout=30000)

    for destination, expected in [
        ("Batch Screening", "Upload CSV, TXT or Excel file containing SMILES"),
        ("Model & Methods", "Model & Methods"),
        ("Molecular Prediction", "Molecular editor"),
        ("Batch Screening", "Upload CSV, TXT or Excel file containing SMILES"),
    ]:
        page.get_by_role("button", name="Open navigation menu").click()
        link = page.get_by_role("link", name=destination, exact=True)
        expect(link).to_be_visible()
        link.click()
        if destination == "Model & Methods":
            expect(page.get_by_role("heading", name=expected, exact=True)).to_be_visible(timeout=15000)
        else:
            expect(page.get_by_text(expected, exact=True)).to_be_visible(timeout=15000)
        expect(page.get_by_role("link", name=destination, exact=True)).to_have_count(0)
        expect(page.get_by_role("button", name="Open navigation menu")).to_be_visible()



@pytest.mark.parametrize("extension", ["CSV", "TXT"])
def test_example_files_download_from_batch_page(page, app_url, extension):
    """Samples are available without preparing a dataset externally."""
    page.goto(app_url + "?view=batch")
    label = f"Download example .{extension}"
    button = page.get_by_role("button", name=label)
    expect(button).to_be_visible(timeout=30000)
    with page.expect_download() as event:
        button.click()
    assert event.value.suggested_filename == f"PanURA_example.{extension.lower()}"


def test_upload_example_txt_shows_smiles_column(page, app_url):
    page.goto(app_url + "?view=batch")
    uploader = page.locator('input[type="file"]')
    expect(page.get_by_text("Upload CSV, TXT or Excel file containing SMILES")).to_be_visible(timeout=30000)
    uploader.set_input_files(str(ROOT / "examples" / "PanURA_example.txt"))
    expect(page.get_by_text("SMILES column")).to_be_visible(timeout=15000)
    expect(page.get_by_role("button", name="Run Screening")).to_be_visible()
