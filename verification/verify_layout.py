
import time
from playwright.sync_api import sync_playwright

def verify_layout():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # Go to home page
        try:
            page.goto("http://localhost:5000")
        except Exception as e:
            print(f"Failed to load page: {e}")
            return

        # Inject dummy content to simulate a result
        page.evaluate("""() => {
            document.getElementById('result').style.display = 'block';

            // Switch to explanation tab
            document.getElementById('explanationTab').click();

            // Add dummy images
            const dummySrc = 'data:image/svg+xml;charset=UTF-8,%3Csvg%20width%3D%22800%22%20height%3D%22400%22%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%20viewBox%3D%220%200%20800%20400%22%20preserveAspectRatio%3D%22none%22%3E%3Crect%20width%3D%22800%22%20height%3D%22400%22%20fill%3D%22%23EEEEEE%22%2F%3E%3Ctext%20x%3D%2250%25%22%20y%3D%2250%25%22%20dominant-baseline%3D%22middle%22%20text-anchor%3D%22middle%22%20fill%3D%22%23AAAAAA%22%20font-size%3D%2240%22%3EDummy%20Explanation%3C%2Ftext%3E%3C%2Fsvg%3E';

            document.getElementById('limeImage').src = dummySrc;
            document.getElementById('shapImage').src = dummySrc;
        }""")

        # Wait for animations/rendering
        time.sleep(1)

        # Take screenshot of the Explanation Section
        result_loc = page.locator("#explanationContent")
        result_loc.scroll_into_view_if_needed()

        # Take full page screenshot just in case
        page.screenshot(path="verification/layout_check.png", full_page=True)
        print("Screenshot saved to verification/layout_check.png")

        browser.close()

if __name__ == "__main__":
    verify_layout()
