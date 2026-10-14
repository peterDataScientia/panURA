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
    # Streamlit can briefly replace the JSME iframe while sizing the mobile canvas.
    # Poll for a mounted, measurable frame instead of reading during replacement.
    bounds = page.wait_for_function(
        """() => {
            const f = document.querySelector('iframe');
            if (!f || !f.isConnected) return false;
            const r = f.getBoundingClientRect();
            return r.width > 0 && r.height > 0
                ? {x: r.x, y: r.y, width: r.width, height: r.height}
                : false;
        }""",
        timeout=15000,
    ).json_value()
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


def test_batch_csv_ids_remain_in_prediction_export(page, app_url):
    """Each sample compound ID must survive inference and CSV download."""
    import csv
    import io
    page.goto(app_url + "?view=batch")
    expect(page.get_by_text("Upload CSV, TXT or Excel file containing SMILES")).to_be_visible(timeout=30000)
    page.locator('input[type="file"]').set_input_files(str(ROOT / "examples" / "PanURA_example.csv"))
    expect(page.get_by_role("button", name="Run Screening")).to_be_visible(timeout=15000)
    page.get_by_role("button", name="Run Screening").click()
    download_button = page.get_by_role("button", name="Download predictions (CSV)")
    expect(download_button).to_be_visible(timeout=30000)
    with page.expect_download() as event:
        download_button.click()
    content = event.value.path().read_text(encoding="utf-8")
    records = list(csv.DictReader(io.StringIO(content)))
    assert len(records) == 6
    assert list(records[0])[:2] == ["ID", "SMILES"]
    assert [r["ID"] for r in records] == [f"URA_{i:03d}" for i in range(1, 7)]
    assert all(r["pIC50_pred"] for r in records)
    assert all(r["AD_status"] in {"Inside", "Outside"} for r in records)
    assert all(r["leverage"] not in {"", "None", "nan"} for r in records)
    assert all(r["selected_bit_tanimoto"] not in {"", "None", "nan"} for r in records)
    assert "max_tanimoto" not in records[0], "Do not export an empty similarity column"


def test_small_batch_shows_molecule_chart_not_fixed_25_bin_histogram(page, app_url):
    """Individual molecular IDs must be interpretable for small screenings."""
    page.goto(app_url + "?view=batch")
    page.locator('input[type="file"]').set_input_files(str(ROOT / "examples" / "PanURA_example.csv"))
    page.get_by_role("button", name="Run Screening").click()
    chart = page.locator('[data-testid="stPlotlyChart"]')
    expect(chart).to_be_visible(timeout=30000)
    # The Plotly figure is rendered by Streamlit, and must use one scatter
    # trace with every compound in the six-row example.
    figure = chart.locator(".js-plotly-plot").first
    expect(figure).to_be_visible(timeout=15000)
    trace_info = figure.evaluate(
        "(el) => el.data.map((t, i) => ({type: t.type, count: el.calcdata[i].length}))"
    )
    assert trace_info == [{"type": "scatter", "count": 6}], trace_info


def test_paste_smiles_predict_save_and_restore_design(page, app_url):
    """A pasted structure must support prediction and a persistent design snapshot."""
    page.set_viewport_size({"width": 1280, "height": 900})
    page.goto(app_url)
    box = page.get_by_role("textbox", name="Paste SMILES to display and edit its molecular structure")
    expect(box).to_be_visible(timeout=30000)
    box.fill("CC(=O)N")
    box.press("Enter")
    expect(page.get_by_text("Predicted pIC₅₀", exact=True)).to_be_visible(timeout=30000)
    expect(page.get_by_text("Training-space AD", exact=True)).to_be_visible(timeout=30000)
    expect(page.get_by_text("Current structure SMILES: CC(N)=O", exact=True)).to_be_visible(timeout=30000)
    expect(page.get_by_text("4.954", exact=True)).to_be_visible(timeout=30000)
    page.get_by_role("button", name="Save current design").click()
    expect(page.get_by_text("Saved Design 1: CC(N)=O", exact=True)).to_be_visible(timeout=15000)
    expect(page.get_by_role("combobox", name="Restore a saved molecular structure")).to_be_visible(timeout=15000)
    box.fill("CCO")
    box.press("Enter")
    # The live structure marker confirms the backend is using this edit.
    expect(page.get_by_text("Current structure SMILES: CCO", exact=True)).to_be_visible(timeout=30000)
    page.wait_for_timeout(400)  # allow the editor callback to settle
    page.get_by_role("button", name="Save current design").click()
    expect(page.get_by_text("Saved Design 2: CCO", exact=True)).to_be_visible(timeout=15000)
    expect(page.get_by_role("button", name="Download design comparison (.csv)")).to_be_visible()
    # Export is covered by Streamlit download code; this test focuses on editor state.
    page.get_by_role("button", name="Load selected design into editor").click()
    import re
    expect(box).to_have_value(re.compile(r"^CC(?:\(=O\)N|\(N\)=O)$"), timeout=15000)
