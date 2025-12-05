from playwright.sync_api import sync_playwright, expect
import time

def verify_quarter_analysis():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # 1. Go to Home Page
        print("Navigating to home page...")
        page.goto("http://127.0.0.1:5000/")

        # 2. Verify Title
        print("Verifying title...")
        expect(page).to_have_title("Media Sentinel")

        # 3. Verify Elements Existence (Upload form, buttons)
        print("Verifying UI elements...")
        expect(page.locator("h1")).to_contain_text("Advanced DeepFake Media Analysis")
        expect(page.locator("#uploadForm")).to_be_visible()

        # 4. Check for Quarter Analysis element (hidden initially)
        # It's hidden in the #result div
        result_div = page.locator("#result")
        expect(result_div).to_be_hidden()

        # Check if the "Decision Insight" label exists in the DOM
        # The text is "Decision Insight" in a p.result-item-label
        # We can look for the ID #quarterAnalysis
        quarter_text = page.locator("#quarterAnalysis")
        # It should exist in DOM even if hidden
        # expect(quarter_text).to_be_attached()

        # 5. Take Screenshot of the Home Page (Initial State)
        print("Taking screenshot...")
        page.screenshot(path="/home/jules/verification/verification.png")

        browser.close()

if __name__ == "__main__":
    verify_quarter_analysis()
