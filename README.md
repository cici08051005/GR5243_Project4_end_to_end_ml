# STAT GR5243 Project 4: Predicting High-Occupancy Airbnb Listings in NYC

## Project Overview

This repository contains our final project for **STAT GR5243: Advanced Data Science and Artificial Intelligence**.

The project follows an end-to-end machine learning workflow using Inside Airbnb New York City data. Our goal is to understand what listing, host, location, and property-related factors are associated with higher Airbnb booking activity, and to build supervised learning models that predict whether a listing is likely to be high-occupancy.

## Research Question

**What listing, host, location, and review-related factors are associated with higher Airbnb booking activity in New York City, and how accurately can we predict high-occupancy listings using machine learning models?**

## Data Source

The raw data come from **Inside Airbnb**, using the New York City listings dataset.

Raw files used in this project include:

- `listings.csv.gz`: detailed listings data
- `listings.csv`: summary listings data
- `neighbourhoods.csv`: neighbourhood information
- `neighbourhoods.geojson`: neighbourhood geographic boundary file

The detailed listings file contains **36,445 rows and 85 columns**.

## Project Motivation

Our initial project idea was to predict Airbnb listing prices in New York City. However, after examining the downloaded raw data, we found that the `price` column was completely missing in both the detailed and summary listings files.

Because of this data quality issue, we reframed the project as a classification problem: predicting whether a listing is high-occupancy.

We define a listing as **high-occupancy** if:

```text
estimated_occupancy_l365d >= 60 ‘’‘

This creates a binary target variable:

’‘’
high_occupancy = 1 if estimated_occupancy_l365d >= 60
high_occupancy = 0 otherwise
‘’‘

Approximately 27.76% of listings are classified as high-occupancy.
