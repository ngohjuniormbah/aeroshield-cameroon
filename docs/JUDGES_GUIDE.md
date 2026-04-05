# Judges Guide

## Project name
**AeroShield Cameroon**

## One-line pitch
AeroShield Cameroon is a climate-driven virtual air-quality sensing and forecasting platform that predicts next-day pollution risk for cities across Cameroon using meteorological data and explainable AI.

## Problem
Air pollution risk changes across cities, seasons, and weather conditions. In many places, physical air-quality sensors are expensive or unavailable. That makes preventive action difficult.

## Our answer
We use the provided national weather dataset to create a **virtual sensor network**. The system estimates an **Air Quality Risk Index (AQRI)** and forecasts the next day's risk level for each city.

## Why it matters
- low-cost alternative where sensors are scarce
- scalable across all regions of Cameroon
- usable for public health, schools, municipalities, and communities
- transforms climate data into operational alerts

## What the AI does
1. Reads the national meteorological dataset
2. Repairs dataset formatting issues safely
3. Engineers spatial, temporal, lag, and rolling features
4. Builds a transparent weather-based AQRI
5. Trains an ML model to predict **next-day AQRI**
6. Exposes outputs through API + dashboard

## What the dashboard shows
- map of predicted next-day risk by city
- high-risk city ranking
- climate vs risk analysis charts
- alert panel for decision support

## Why this is innovative
This project reframes air-quality monitoring as a **virtual sensing problem**.
Instead of needing hardware everywhere, we infer risk using weather intelligence and historical patterns.

## What makes it strong for this competition
- directly aligned with climate + health resilience
- uses the official dataset as the technical core
- includes a real model, backend, dashboard, and deployment path
- easy to explain, useful to local decision-makers, and expandable later with real PM2.5 data

## What to say during the demo
"We built AeroShield Cameroon as a virtual sensor network for air-quality risk. Since physical monitors are limited, our platform uses weather data, lagged climate signals, and machine learning to forecast where tomorrow's air-risk is likely to rise. The map shows which cities need attention, the charts explain why, and the alert system translates analytics into action."

## Likely judge questions and how to answer them

### Why not use hardware?
Because the competition is software and data driven. Our innovation is showing that climate data can create a scalable low-cost monitoring layer even where hardware coverage is weak.

### Why use a proxy target?
The baseline dataset does not provide measured PM2.5 directly. We therefore built a transparent AQRI from known meteorological aggravating factors, then forecast its next-day behavior. This respects the dataset while still solving the problem in a practical way.

### Can this become real?
Yes. The system is designed so that measured air-quality data can later be added to calibrate or replace the proxy target. The architecture already supports that upgrade.

### Who would use it?
Public health teams, municipal planners, schools, hospitals, NGOs, and climate resilience programs.
