# Mobile Price and Range Prediction

This project predicts the market price and price category of a mobile phone based on its technical specifications using machine learning.

Developed during internship at Unified Mentorship Pvt. Ltd.

---

## Objective

Build a machine learning model that can:

- Predict the realistic price (in ₹) of a mobile phone
- Classify the phone into a price range:
  - 0 = Low (₹2,000 – ₹9,000)
  - 1 = Medium (₹9,100 – ₹20,000)
  - 2 = High (₹21,000 – ₹60,000)
  - 3 = Very High (₹61,000 – ₹1,50,000)

---

## Dataset

- Source: `mobile_price_dataset_realistic_cleaned.csv`
- Contains over 2000 mobile entries with 21+ features like RAM, battery, camera, screen size, etc.
- Preprocessed:
  - Removed outdated fields (e.g., `three_g`)
  - One-hot encoded network features (2G to 5G)
  - Added realistic price column
  - Balanced classification target

---


