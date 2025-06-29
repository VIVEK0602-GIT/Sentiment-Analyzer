import os
import time
import sys
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

def take_screenshot(url, output_path, wait_time=5, width=1280, height=800):
    """
    Take a screenshot of a webpage using Selenium and Chrome
    
    Parameters:
    -----------
    url : str
        URL of the webpage to capture
    output_path : str
        Path to save the screenshot
    wait_time : int
        Time to wait for the page to load in seconds
    width : int
        Browser window width
    height : int
        Browser window height
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Configure Chrome options
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument(f"--window-size={width},{height}")
        
        # Create a new Chrome driver
        driver = webdriver.Chrome(options=chrome_options)
        
        # Navigate to the URL
        print(f"Navigating to {url}")
        driver.get(url)
        
        # Wait for the page to load
        print(f"Waiting {wait_time} seconds for page to load...")
        time.sleep(wait_time)
        
        # Take screenshot
        print(f"Taking screenshot and saving to {output_path}")
        driver.save_screenshot(output_path)
        
        # Close the driver
        driver.quit()
        
        print("Screenshot captured successfully")
        return True
    
    except Exception as e:
        print(f"Error capturing screenshot: {e}")
        return False

def main():
    """Main function to run the screenshot capture"""
    # Check if Selenium and Chrome are available
    try:
        from selenium import webdriver
    except ImportError:
        print("Selenium is not installed. Please install it using: pip install selenium")
        print("Also make sure you have Chrome and ChromeDriver installed")
        return False
    
    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    
    # Take screenshots of our dashboards
    screenshots = [
        {
            "url": "http://localhost:8501",  # Default Streamlit port
            "output_path": "results/streamlit_dashboard_screenshot.png",
            "wait_time": 5,
            "description": "Streamlit Dashboard"
        },
        {
            "url": "file://" + os.path.abspath("results/vader_dashboard.html"),
            "output_path": "results/vader_dashboard_screenshot.png",
            "wait_time": 2,
            "description": "Simple HTML Dashboard"
        }
    ]
    
    success = True
    
    for screenshot in screenshots:
        print(f"\nCapturing {screenshot['description']}...")
        result = take_screenshot(
            screenshot["url"],
            screenshot["output_path"],
            screenshot["wait_time"]
        )
        if not result:
            success = False
    
    if success:
        print("\nAll screenshots captured successfully!")
        print("You can now include them in your README file.")
    else:
        print("\nSome screenshots could not be captured.")
        print("Please make sure the services are running before taking screenshots.")

if __name__ == "__main__":
    main() 