# grandlyon-photo-clusters

## Project Description
This project aims to analyze a dataset of photos from the Grand Lyon area to automatically identify and characterize Points of Interest (POIs) and events.

The workflow consists of:
1.  **Data Processing**: Ingestion, cleaning, and filtering of raw photo data (geolocation, timestamps, text).
2.  **Clustering**: Applying spatial clustering algorithms (e.g., DBSCAN, K-Means) to grouping photos into areas of interest.
3.  **Text Mining**: Using methods like TF-IDF and Association Rules to generate descriptive labels for each cluster based on user tags and photo titles.
4.  **Temporal Analysis**: Distinguishing between permanent landmarks and temporary events by analyzing the temporal distribution of photos within clusters.
5.  **Visualization**: An interactive map application to display the results and explore the discovered clusters.