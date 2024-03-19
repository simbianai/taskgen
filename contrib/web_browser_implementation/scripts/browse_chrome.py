from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Create a new instance of the Chrome driver
driver = webdriver.Chrome()

# Navigate to a web page
driver.get("https://www.example.com")

try:
    # Wait for the presence of a specific element
    element = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.ID, "myElement"))
    )
    # Perform actions on the element
    element.click()

    # Fill in a form field
    input_field = driver.find_element(By.NAME, "username")
    input_field.send_keys("myusername")

    # Submit the form
    submit_button = driver.find_element(By.CSS_SELECTOR, "button[type='submit']")
    submit_button.click()

    # Extract data from the page
    result = driver.find_element(By.CLASS_NAME, "result").text
    print("Result:", result)

finally:
    # Close the browser
    driver.quit()