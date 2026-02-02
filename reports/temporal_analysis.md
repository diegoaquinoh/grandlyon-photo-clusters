# Temporal Analysis Report

**Generated:** 2026-02-02 09:32:24  
**Total Clusters Analyzed:** 226  
**Total Photos:** 95,904.0

---

## Summary

This analysis classifies photo clusters based on their temporal patterns to distinguish between:

- **Permanent POIs**: Consistent year-round activity (e.g., landmarks, museums)
- **Recurring Events**: Seasonal patterns (e.g., Fête des Lumières, summer tourism)
- **One-time Events**: Single spike of activity (e.g., concerts, sports events)

### Classification Results

| Type | Clusters | Photos | % of Photos |
|------|----------|--------|-------------|
| **Permanent POI** | 104 | 51,590.0 | 53.8% |
| **Recurring Event** | 84 | 33,389.0 | 34.8% |
| **One-time Event** | 38 | 10,925.0 | 11.4% |

---

## Key Findings

### December Spikes (Fête des Lumières Candidates)

The following clusters show strong December activity:

| Cluster | December Ratio | Total Photos |
|---------|---------------|--------------|
| 41.0 | 95.9% | 221.0 |
| 30.0 | 94.5% | 128.0 |
| 222.0 | 85.8% | 289.0 |
| 93.0 | 83.9% | 217.0 |
| 182.0 | 80.0% | 170.0 |
| 185.0 | 76.3% | 207.0 |
| 118.0 | 67.6% | 210.0 |
| 20.0 | 66.8% | 253.0 |
| 211.0 | 65.3% | 360.0 |
| 193.0 | 63.4% | 183.0 |

### Summer Peaks (Tourism/Festival Candidates)

The following clusters show strong July-August activity:

| Cluster | Summer Ratio | Total Photos |
|---------|-------------|--------------|
| 40.0 | 100.0% | 180.0 |
| 39.0 | 100.0% | 351.0 |
| 0.0 | 96.3% | 914.0 |
| 153.0 | 89.1% | 129.0 |
| 5.0 | 88.7% | 142.0 |
| 16.0 | 79.7% | 197.0 |
| 148.0 | 77.0% | 161.0 |
| 105.0 | 74.9% | 171.0 |
| 178.0 | 70.9% | 151.0 |
| 119.0 | 69.0% | 155.0 |

---

## Visualizations

### Cluster × Month Heatmap

The heatmap below shows the normalized monthly distribution for the top 30 clusters:

![Temporal Heatmap](temporal_heatmap.png)

### Classification Summary

![Classification Summary](temporal_classification_summary.png)

### Time Series: Top 10 Clusters

![Time Series](temporal_timeseries.png)

### Monthly Distribution by Type

![Monthly Distribution](temporal_monthly_distribution.png)

---

## Methodology

### Classification Rules

1. **Recurring Events** are identified when:
   - December ratio > 25% (Fête des Lumières pattern)
   - Summer ratio > 35% (tourism pattern)
   - High coefficient of variation (CV > 1.0) with clear peak

2. **One-time Events** are identified when:
   - Peak month contains > 50% of photos
   - Activity in 3 or fewer months

3. **Permanent POIs** are the default when:
   - Coefficient of variation < 0.8
   - Activity in 6+ months
   - Multi-year presence (5+ years)

### Peak Detection

Z-score based peak detection identifies months with activity more than 1.5 standard deviations above the mean.

---

## Known Lyon Events Reference

| Event | Months | Description |
|-------|--------|-------------|
| Fête des Lumières | December | Annual light festival |
| Nuits de Fourvière | June-July | Summer arts festival |
| Biennale de la Danse | September | Biennial dance festival |
| Quais du Polar | April | Crime fiction festival |
| Summer Tourism | July-August | Peak tourist season |
