# Delhi Pollution AI System

AI-driven decision support system for analyzing, explaining, and simulating air pollution (AQI) in Delhi.

---

## Overview

This system moves beyond traditional AQI dashboards.

Instead of only visualizing pollution levels, it enables structured analysis and decision-making through:

- Explainability using SHAP  
- Scenario simulation through feature manipulation  
- Knowledge-grounded reasoning using LLMs  

The goal is to bridge the gap between data analysis and actionable environmental decisions.

---

## Core Framework

The system is designed as:

    Analysis → Explanation → Simulation → AI Reasoning

This enables users to understand not just what is happening, but why and what can be done.

---

## Problem Statement

### Traditional AQI Systems
- Static visualization dashboards  
- Limited interpretability  
- No simulation capability  
- No actionable insights  

### Proposed System
- Identifies key pollutant drivers  
- Simulates policy interventions  
- Provides interpretable insights  
- Supports data-driven decision-making  

---

## System Components

- Model: XGBoost (trained on pollutant features)  
- Explainability: SHAP  
- Simulation Engine: Feature perturbation-based scenario testing  
- Knowledge Layer: Local knowledge base retrieval  
- LLM Integration: Context-aware insight generation  
- Interface: Streamlit  

---

## Use Cases

### Government and Urban Planning
- Policy impact assessment  
- Pollution control strategy planning  

### Environmental Consulting
- Scenario-based analysis  
- Compliance and reporting support  

### Industrial and Energy Sector
- Emission source analysis  
- Operational optimization  

---

## Practical Impact

- Improves understanding of pollution drivers  
- Supports evidence-based interventions  
- Enables structured decision-making  

---

## Setup

```bash
pip install -r requirements.txt